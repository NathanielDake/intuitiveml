from collections import deque


class DfsIter:

    def __init__(self, g):
        self.g = g
        self.visited = {node: False for node in g.vertices()}

    def dfs(self):
        v = list(self.g.vertices())[0]

        stack = deque()
        stack.append(v)

        while len(stack) > 0:
            v = stack.pop()
            if not self.visited[v]:
                self.visited[v] = True
                neighbors = [x for x in self.g._outgoing[v].keys()]
                for neighbor in neighbors:
                    stack.append(neighbor)
