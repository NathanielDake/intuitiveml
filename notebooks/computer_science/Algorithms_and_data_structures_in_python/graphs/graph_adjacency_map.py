class Graph:
    """Representation of a simple graph using an adjacency map."""

    def __init__(self, directed=False):
        """Create an empty graph (undirected, by default)"""

        self._outgoing = {}
        # Only create a second map for directed graph; for undirected, use alias
        self._incoming = {} if directed else self._outgoing

    def is_directed(self):
        """Return true if this is a directed graph. False if undirected
            Note: Based on initial declaration, not graphs contents.
        """
        return self._incoming is not self._outgoing  # directed if maps are distinct

    def vertex_count(self):
        """Return the number of vertices in the graph"""
        return len(self._outgoing)

    def vertices(self):
        """Return an iteration of all vertices of the graph."""
        return self._outgoing.keys()

    def edge_count(self):
        """Return the number of edges in the graph"""
        total = sum(len(self._outgoing[v] for v in self._outgoing))
        return total if self.is_directed() else total // 2

    def edges(self):
        """Return a set of all edges of the graph"""
        result = set()
        for secondary_map in self._outgoing.values():
            result.update(secondary_map.values())  # add edges to resulting set
        return result

    def get_edge(self, u, v):
        """Return the edge from u to v, or None if not adjacent."""
        return self._outgoing[u].get(v, None)

    def degree(self, v, outgoing=True):
        """Return number of outgoing edges incident to vertex v in the graph.

        If graph is directed, optional parameter used to count incoming edges"""
        adj = self._outgoing if outgoing else self._incoming
        return len(adj[v])

    def incident_edges(self, v, outgoing=True):
        """Return all (outgoing) edges incident to vertex v in the graph

        If graph is directed, optional parameter used to request incoming edges.
        """
        adj = self._outgoing if outgoing else self._incoming
        for edge in adj[v].values():
            yield edge

    def insert_vertex(self, x=None):
        """Insert and return a new Vertex with element x."""
        v = self.Vertex(x)
        self._outgoing[v] = {}
        if self.is_directed():
            self._incoming[v] = {}  # need distinct map for incoming edges
        return v

    def insert_edge(self, u, v, x=None):
        """Insert and return a new Edge from u to v with auxiliary element x."""
        e = self.Edge(u, v, x)
        self._outgoing[u][v] = e
        self._incoming[v][u] = e
        return

    class Vertex:
        """Lightweight vertex structure for a graph"""
        __slots__ = '_element'

        def __init__(self, x):
            """Do not call constructor directly. Use Graphs insert_vertex(x)."""
            self._element = x

        def element(self):
            """Return element associated with this vertex."""
            return self._element

        def __hash__(self):
            return hash(id(self))  # will allow vertex to be a map/set key

    class Edge:
        """Lightweight edge structure for a graph."""
        __slots__ = '_origin', '_destination', '_element'

        def __init__(self, u, v, x):
            """Do not call constructor directly. Use Graph's insert_index(u, v, x)."""
            self._origin = u
            self._destination = v
            self._element = x

        def endpoints(self):
            """Return (u, v) tuple for vertices u and v."""
            return (self._origin, self._destination)

        def opposite(self, v):
            """Return the vertex that is opposite v on this edge."""
            return self._destination if v is self._origin else self._origin

        def element(self):
            """Return element associated with this edge."""
            return self._element

        def __hash__(self):  # will allow edge to be a map/set key
            return hash((self._origin, self._destination))


# Utils
def create_random_graph():
    import random
    from itertools import combinations

    g = Graph()

    nodes = [x for x in "abcdefghijklmnop"]
    for node in nodes:
        g.insert_vertex(node)

    possible_edges = list(combinations(list(g.vertices()), 2))
    actual_edges = random.sample(possible_edges, 30)
    for edge in actual_edges:
        g.insert_edge(edge[0], edge[1])

    return g


def compute_adjacency_matrix(g):
    import pandas as pd

    node_list = pd.Series([node.element() for node in g.vertices()])
    adjacency_mat = pd.DataFrame(index=node_list, columns=node_list)
    for node_i in g.vertices():
        adjacent_nodes_to_node_i = [x._destination.element() for x in list(g.incident_edges(node_i))]
        adjacent_nodes_to_node_i_mask = [1 if node in adjacent_nodes_to_node_i else 0 for node in node_list]
        adjacency_mat = adjacency_mat.assign(**{node_i.element(): adjacent_nodes_to_node_i_mask})
    return adjacency_mat


def plot_graph_via_nx(g):
    import networkx as nx
    import matplotlib.pyplot as plt

    adjacency_matrix = compute_adjacency_matrix(g)
    G = nx.from_pandas_adjacency(adjacency_matrix)
    plt.figure(figsize=(10, 7))
    nx.draw(G, with_labels=True, node_color="orange", node_size=800, font_weight="bold", font_size=20)
