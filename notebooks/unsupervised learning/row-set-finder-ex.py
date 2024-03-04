# %% [markdown]
#

from typing import Any, Generator, Iterable, Optional, Sequence, Union

import datasketch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# %%
import pandas as pd
import pyarrow as pa
import sklearn.datasets


# %%
class DataSet:
    df: pd.DataFrame
    metric: str
    minhash_kwargs: dict[str, Any]

    def __init__(self, df: pd.DataFrame, metric: str, **minhash_kwargs: Any) -> None:
        self.df = df
        self.metric = metric
        self.minhash_kwargs = minhash_kwargs

    def create_mask(self, mask: pd.Series, name: str) -> "Mask":
        return Mask(self, mask, name)

    def create_rowset(self, mask: "Mask") -> "RowSet":
        if mask.ds is not self:
            raise ValueError(
                "Cannot create a RowSet from a Mask that was created from a different DataSet"
            )
        return RowSet.from_mask(self, mask)

    def get_root_rowset(self, name: str = "ROOT") -> "RowSet":
        """Get the RowSet that contains all rows in the dataset"""
        mask = self.create_mask(~self.df.isna().any(axis=1), name)
        return self.create_rowset(mask)


class Mask:
    ds: DataSet
    name: str
    mask: pd.Series

    def __init__(self, ds: DataSet, mask: pd.Series, name: str = "Unnamed"):
        if name is None:
            name = str(mask.name)
        mask.name = name
        self.name = name
        self.mask = mask
        self.ds = ds

    def __repr__(self):
        return f"Mask({self.name})"

    def __or__(self, other: "Mask") -> "Mask":
        if self.ds is not other.ds:
            raise ValueError("Cannot combine masks from different DataSets")
        return Mask(self.ds, self.mask | other.mask, f"({self.name} | {other.name})")

    def __and__(self, other: "Mask") -> "Mask":
        if self.ds is not other.ds:
            raise ValueError("Cannot combine masks from different DataSets")
        return Mask(self.ds, self.mask & other.mask, f"({self.name} & {other.name})")


class RowSet:
    """A class representing a set of rows in a DataFrame.
    A MinHash sketch is used to represent the set of rows. This is a dense fixed size vector.
    A CDF is used to represent the distribution of a metric across the rows.
    """

    # TODO: support multiple metrics
    ds: DataSet
    mask: Mask
    sketch: datasketch.MinHash
    metric_quantiles: pd.Series
    metric_cdf: pd.Series
    size: int

    _default_cdf_qs = np.linspace(0, 1, 100)

    def __init__(
        self,
        ds: DataSet,
        size: int,
        sketch: datasketch.MinHash,
        metric_quantiles: pd.Series,
        mask: Mask,
    ):
        self.ds = ds
        self.size = size
        self.sketch = sketch
        self.metric_quantiles = metric_quantiles
        self.mask = mask

        self.metric_cdf = (
            metric_quantiles.reset_index()
            .groupby(metric_quantiles.name)[metric_quantiles.index.name]
            .max()
        )

    def __len__(self) -> int:
        return self.size

    def __repr__(self):
        return f"RowSet(size={self.size}, mask={self.mask.name})"

    def __or__(self, other: "RowSet") -> "RowSet":
        """Compute the union of two RowSets"""
        if self.ds is not other.ds:
            raise ValueError("Cannot combine RowSets from different DataSets")
        new_mask = self.mask | other.mask
        new_rowset = RowSet.from_mask(self.ds, new_mask)
        return new_rowset

    def __and__(self, other: "RowSet") -> "RowSet":
        """Compute the intersection of two RowSets"""
        if self.ds is not other.ds:
            raise ValueError("Cannot combine RowSets from different DataSets")
        new_mask = self.mask & other.mask
        new_rowset = RowSet.from_mask(self.ds, new_mask)
        return new_rowset

    @classmethod
    def from_mask(cls, ds: DataSet, mask: Mask, cdf_qs: Sequence[float] = None) -> "RowSet":
        masked_df = ds.df[mask.mask]
        if len(masked_df) == 0:
            raise ValueError("Mask must have at least one True value")
        size = len(masked_df)

        # build the index sketch
        sketch = datasketch.MinHash(**ds.minhash_kwargs)
        idx = masked_df.index
        for i in idx:
            sketch.update(bytes(i))

        # build the metric cdf
        if cdf_qs is None:
            cdf_qs = cls._default_cdf_qs
        cdf_vals = pa.compute.tdigest(pa.array(masked_df[ds.metric].values), q=cdf_qs)
        metric_quantiles = pd.Series(
            cdf_vals, index=pd.Index(cdf_qs, name="quantile"), name=ds.metric
        )
        return cls(ds, size=size, sketch=sketch, metric_quantiles=metric_quantiles, mask=mask)


# %%
class RowSetCollection:
    """Helper class to build a MinHashLSH index from multiple RowSets
    This will allow us to do a single LSH query to find the closest RowSet to a given RowSet
    We can then build a graph of RowSets where the edges are the similarity between RowSets
    This will allow us find cliques of similar RowSets which we can then use to represent
    a single "concept" or "cluster" of data.
    """

    rowsets: dict[str, RowSet]
    index: datasketch.MinHashLSH # This is not used while finding rowsets, but after we have found them!
    G: Optional[nx.Graph]

    def __init__(self, rowsets: Sequence[RowSet], index: datasketch.MinHashLSH) -> None:
        self.rowsets = {}
        # build the LSH index
        self.index = index
        self.insert(rowsets)

    def insert(self, obj: Union[RowSet, Sequence[RowSet]]) -> None:
        if isinstance(obj, RowSet):
            self._insert(obj)
        else:
            for rs in obj:
                self._insert(rs)

    def _insert(self, rowset: RowSet) -> None:
        if rowset.mask.name in self.rowsets:
            raise ValueError(f"RowSet {rowset.mask.name} already exists")
        self.rowsets[rowset.mask.name] = rowset
        self.index.insert(rowset.mask.name, rowset.sketch)
        self.G = None

    def __len__(self) -> int:
        return len(self.rowsets)

    def __iter__(self) -> Generator[str, None, None]:
        return (x for x in self.rowsets.keys())

    def __next__(self) -> Iterable[str]:
        names = iter(self.rowsets.keys())
        for name in names:
            yield name

    def __repr__(self) -> str:
        return f"RowSetCollection({len(self.rowsets)} rowsets)"

    def __getitem__(self, key: str) -> RowSet:
        return self.rowsets[key]

    def query(self, rowset: RowSet) -> dict[str, RowSet]:
        neighbors = self.index.query(rowset.sketch)
        return {rs_name: self.rowsets[rs_name] for rs_name in neighbors}

    def to_nx(self) -> nx.Graph:
        if self.G is not None:
            return self.G
        g = nx.Graph()
        for n, rs in self.rowsets.items():
            g.add_node(n)
            for rs2 in self.query(rs):
                if rs2 != n:
                    g.add_edge(n, rs2)
        self.G = g
        return g

    def query_cliques(self, id_: str) -> list[dict[str, RowSet]]:
        rowset = self.rowsets[id_]
        G = self.to_nx()
        cliques = nx.find_cliques(G, nodes=[rowset.mask.name])
        return [{rs_name: self.rowsets[rs_name] for rs_name in c} for c in cliques]


def to_lsh_ensemble(
    rowsets: dict[str, "RowSet"], threshold: float, **kwargs: Any
) -> datasketch.MinHashLSHEnsemble:
    lsh = datasketch.MinHashLSHEnsemble(threshold=threshold, **kwargs)
    entries = [(rs.mask.name, rs.sketch, rs.size) for rs in rowsets.values()]
    lsh.index(entries)
    return lsh


def get_best_rowset_candidates(
    candidates: dict[str, "RowSet"],
    rowset: "RowSet",
    min_threshold: float = 0.1,
    max_threshold: float = 0.9,
) -> list["RowSet"]:
    """Find the RowSets that are within a range of containment for the given RowSet"""
    min_lsh_ensemble = to_lsh_ensemble(candidates, min_threshold, num_perm=128 * 2)
    max_lsh_ensemble = to_lsh_ensemble(candidates, max_threshold, num_perm=128 * 2)

    min_set = set(min_lsh_ensemble.query(rowset.sketch, rowset.size))
    max_set = set(max_lsh_ensemble.query(rowset.sketch, rowset.size))
    # The min set is a super-set of the max set, we want to remove the max set from the min set
    # to get the set of RowSets that are within the range of containment
    return [candidates[rowset] for rowset in min_set if rowset not in max_set]


def build_column_rowsets(ds: DataSet) -> dict[str, "RowSet"]:
    """Build a set of RowSets for each column in a dataset"""
    rowsets = {}
    for col in ds.df.columns:
        if col == ds.metric:
            continue
        for val in ds.df[col].unique():
            mask = ds.create_mask(ds.df[col] == val, f"{col} = {val}")
            rowset = ds.create_rowset(mask)
            rowsets[rowset.mask.name] = rowset
    return rowsets


def max_cdf_diff(rs1: RowSet, rs2: RowSet) -> float:
    # function to calculate the max difference in the cumulative distribution functions of two rowsets

    # since the index values can be different for the two cdfs, we need to align them
    # we do this by taking the union of the index values and then filling in the missing values with
    # the nearest value
    cdf1 = rs1.metric_cdf
    cdf2 = rs2.metric_cdf
    cdf1 = cdf1.reindex(cdf1.index.union(cdf2.index), method="nearest")
    cdf2 = cdf2.reindex(cdf2.index.union(cdf1.index), method="nearest")
    # now we can calculate the max difference between the two cdfs
    return np.abs(cdf1 - cdf2).max()


class RowSetFinder:
    atom_rowsets: dict[str, RowSet]

    def __init__(
        self,
        dataset: DataSet,
        min_threshold: float = 0.9,
        max_threshold: float = 0.1,
        max_rowsets: int = 100,
        max_candidates: int = 10,
        min_size: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.max_rowsets = max_rowsets
        self.max_candidates = max_candidates
        self.min_size = min_size if min_size is not None else dataset.df.shape[0] // 50

        # initialize the atom rowsets
        self._build_atom_rowsets()

    def _build_atom_rowsets(self) -> None:
        self.atom_rowsets = build_column_rowsets(self.dataset)

    def next_rowsets(self, rowset: RowSet, rowset_lineage: dict[str, set[str]]) -> list[RowSet]:
        # first find the candidates that are within the range of containment
        candidate_atoms = get_best_rowset_candidates(
            self.atom_rowsets, rowset, self.min_threshold, self.max_threshold
        )
        # for each candidate, we need to union/interset with the rowset to get a new rowset
        candidates = []
        for atom in candidate_atoms:
            if atom.mask.name in rowset_lineage[rowset.mask.name]:
                continue
            candidate = rowset & atom
            # filter out the candidates that are too small
            if candidate.size >= self.min_size:
                candidates.append(candidate)
                # update the lineage
                assert candidate.mask.name not in rowset_lineage
                rowset_lineage[candidate.mask.name] = (
                    rowset_lineage[rowset.mask.name].copy().union({atom.mask.name})
                )

        # next, sort the candidates by the max difference in cdf
        candidates = sorted(candidates, key=lambda rs: max_cdf_diff(rowset, rs), reverse=True)
        print(f"Candidates: {len(candidates)} for {rowset.mask.name}")
        print([(x, max_cdf_diff(rowset, x)) for x in candidates])
        # finally, return the top candidates
        return candidates[: self.max_candidates]

    def find_rowsets(self, minhashlsh: datasketch.MinHashLSH) -> RowSetCollection:
        rowsets = RowSetCollection([], minhashlsh)
        rowset_lineage: dict[str, set[str]] = {}

        # first start with the root rowset
        root = self.dataset.get_root_rowset()
        rowsets.insert(root)
        rowset_lineage[root.mask.name] = set()
        queue = [root]

        # then find the sub rowsets for each rowset
        # keep doing this until we have found the max number of rowsets or we have exhausted the queue
        while len(rowsets) < self.max_rowsets and queue:
            rowset = queue.pop()
            next_rowsets = self.next_rowsets(rowset, rowset_lineage)
            rowsets.insert(next_rowsets)
            queue.extend(next_rowsets)
            # reprioritize the rowsets that have the largest size
            queue = sorted(queue, key=lambda rs: rs.size, reverse=True)

        return rowsets


# %%
# load a toy dataset
tmpX, tmpy = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)
# discretize all the X features
tmpX = tmpX.apply(pd.cut, bins=10, labels=False)
# join the X and y dataframes
df = tmpX.join(tmpy.rename("y"))
df.head()

# %%
ds = DataSet(df, "y", num_perm=128 * 2)

set1 = ds.create_rowset(ds.create_mask(df["age"] == 0, "age = 0"))
set2 = ds.create_rowset(ds.create_mask(df["sex"] == 0, "sex = 0"))
set3 = set1 & set2

fig, ax = plt.subplots()
set1.metric_cdf.plot(ax=ax, label=set1.mask.name)
set2.metric_cdf.plot(ax=ax, label=set2.mask.name)
set3.metric_cdf.plot(ax=ax, label=set3.mask.name)
ax.legend()

fig, ax = plt.subplots()
set1.metric_quantiles.plot(ax=ax, label=set1.mask.name)
set2.metric_quantiles.plot(ax=ax, label=set2.mask.name)
set3.metric_quantiles.plot(ax=ax, label=set3.mask.name)
ax.legend()

print(
    set1.sketch.jaccard(set2.sketch),
    set2.sketch.jaccard(set3.sketch),
    set1.sketch.jaccard(set3.sketch),
)

# %%
index = datasketch.MinHashLSH(threshold=0.05, num_perm=128 * 2)
rwc = RowSetCollection([set1, set2, set3], index)
g = rwc.to_nx()
rwc.query_cliques(set1.mask.name)

# %%
atom_rowsets = build_column_rowsets(ds)
len(atom_rowsets)

# %%
candidates = get_best_rowset_candidates(atom_rowsets, set1)
len(candidates)

# %%
root_rowset = ds.get_root_rowset()
candidates = get_best_rowset_candidates(atom_rowsets, root_rowset)
len(candidates)

# %%
finder = RowSetFinder(ds, 0.1, 1.0, max_rowsets=200, max_candidates=50)

minhashlsh = datasketch.MinHashLSH(threshold=0.5, num_perm=128 * 2)
rowsets = finder.find_rowsets(minhashlsh)
len(rowsets)

# %%
list(rowsets.rowsets.values())

# %%
clique_sizes = pd.Series([len(rowsets.query_cliques(x)) for x in rowsets], index=rowsets)
clique_sizes[clique_sizes > 1]

# %%
rowsets.query_cliques(ds.get_root_rowset().mask.name)

# %%
means = []
for rs in rowsets:
    means.append((rs, ds.df[rowsets[rs].mask.mask][ds.metric].mean()))

means = pd.Series([x[1] for x in means], index=[x[0] for x in means])
scaled_means = means / means.loc["ROOT"]

stds = []
for rs in rowsets:
    stds.append((rs, ds.df[rowsets[rs].mask.mask][ds.metric].std()))

stds = pd.Series([x[1] for x in stds], index=[x[0] for x in stds])
scaled_stds = stds / stds.loc["ROOT"]

sizes = pd.Series([len(rowsets[rs]) for rs in rowsets], index=rowsets)

stats = pd.merge(
    scaled_means.to_frame(name="mean"),
    scaled_stds.to_frame("std"),
    left_index=True,
    right_index=True,
    suffixes=("_mean", "_std"),
)
stats = pd.merge(stats, sizes.to_frame("size"), left_index=True, right_index=True)
stats

# %%
tmp = stats.copy()
tmp["score"] = ((1 - stats["mean"]).abs() * (1 - stats["std"]).abs()) * stats["size"]
print(tmp.sort_values(by="score", ascending=False).to_string())

# %%
print(tmp.sort_values(by="mean", ascending=False).to_string())

# %%
fig, ax = plt.subplots(figsize=(12, 12))
ax = pd.plotting.scatter_matrix(stats, ax=ax, diagonal="hist")
fig.show()

# %%
