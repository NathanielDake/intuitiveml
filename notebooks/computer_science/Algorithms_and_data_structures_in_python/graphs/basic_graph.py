from collections import defaultdict


class WeightedGraph:
    def __init__(self, currency_conversion_pairs):
        self.nodes = self._init_nodes(currency_conversion_pairs)
        self.edges = self._init_edges(currency_conversion_pairs)

    def _init_nodes(self, currency_conversion_pairs):
        nodes = set()
        for cp in currency_conversion_pairs:
            nodes.add(cp[0])
            nodes.add(cp[1])
        return nodes

    def _init_edges(self, currency_conversion_pairs):
        edges = defaultdict(dict)
        for cp in currency_conversion_pairs:
            edges[cp[0]].update({cp[1]: cp[2]})
            edges[cp[1]].update({cp[0]: 1 / cp[2]})

        return edges
