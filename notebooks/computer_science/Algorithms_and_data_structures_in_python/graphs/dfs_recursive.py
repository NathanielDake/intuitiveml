class DfsRecursive:

    def __init__(self, g):
        self.g = g
        self.n = g.vertex_count()
        self.visited = {node: False for node in g.vertices()}

    def dfs(self, v):
        self.visited[v] = True
        neighbors = [x for x in self.g._outgoing[v].keys()]
        for neighbor in neighbors:
            if not self.visited[neighbor]:
                self.dfs(neighbor)
        return
