from collections import defaultdict, deque
from queue import Queue
from functools import reduce
from operator import mul


explored = {}  # Explored essentially tracks what nodes have been added to Queue


def bfs_v1a(g, s):
    """Outlined in Algorithms Illuminated (Tim Roughgarden)
    - g: graph
    - s: starting node
    """
    explored = {key: False for key in g.nodes}  
    explored[s] = True

    q = Queue()
    q.put(s)

    while not q.empty():
        v = q.get()
        for w in g.edges[v].keys():
            if not explored[w]:
                explored[w] = True
                q.put(w)
    return explored


def bfs_v1b(g, s):
    """Outlined in Algorithms Illuminated (Tim Roughgarden)
    - g: graph
    - s: starting node

    Several new lines added in order to track the shortest paths.
    """
    explored = {key: False for key in g.nodes}
    explored[s] = True
    shortest_paths = defaultdict(list)
    shortest_paths[s].append(s)

    q = Queue()
    q.put(s)

    while not q.empty():
        v = q.get()
        for w in g.edges[v].keys():
            if not explored[w]:
                explored[w] = True
                shortest_paths[w].extend(shortest_paths[v] + [w])
                q.put(w)
    return explored, shortest_paths


def bfs_v1c(g, s, end):
    """Outlined in Algorithms Illuminated (Tim Roughgarden)
    - g: graph
    - s: starting node
    - w: end node

    Several new lines added in order to track the shortest paths AND the currency value
    """
    explored = {key: False for key in g.nodes}
    explored[s] = True
    shortest_paths = defaultdict(list)
    shortest_paths[s].append(s)

    q = Queue()
    q.put(s)

    while not q.empty():
        v = q.get()
        for w in g.edges[v].keys():
            if not explored[w]:
                explored[w] = True
                shortest_paths[w].extend(shortest_paths[v] + [w])
                q.put(w)

    exchange_product = []
    for i in range(len(shortest_paths[end]) - 1):
        curnode, nextnode = shortest_paths[end][i], shortest_paths[end][i + 1]
        exchange_product.append(g.edges[curnode][nextnode])
    exchange_val = reduce(mul, exchange_product)
    return shortest_paths[end], exchange_val


def bfs_v2a(g, s):
    """Outlined in Algorithm Design Manual (Steve Skiena)"""

    state = {key: "undiscovered" for key in g.nodes}
    state[s] = "discovered"
    parents = {key: None for key in g.nodes}

    q = Queue()
    q.put(s)

    while not q.empty():
        v = q.get()
        # Process this vertex as desired 

        for w in g.edges[v].keys():
            # Process this edge as desired
            if state[w] == "undiscovered":
                state[w] = "discovered"
                parents[w] = v
                q.put(w)
        state[v] = "processed"

    return state, parents


def bfs_v2b(g, s, end):
    """Outlined in Algorithm Design Manual (Steve Skiena)"""

    state = {key: "undiscovered" for key in g.nodes}
    state[s] = "discovered"
    parents = {key: None for key in g.nodes}

    q = Queue()
    q.put(s)

    while not q.empty():
        v = q.get()
        # Process this vertex as desired 

        for w in g.edges[v].keys():
            # Process this edge as desired
            if state[w] == "undiscovered":
                state[w] = "discovered"
                parents[w] = v
                q.put(w)
        state[v] = "processed"

    shortest_path = [end]
    while True:
        shortest_path.append(parents[end])
        if parents[end] == s:
            break
        end = parents[end]

    shortest_path.reverse()

    return state, parents, shortest_path


def bfs_v2c(g, s, end):
    """Outlined in Algorithm Design Manual (Steve Skiena)"""

    state = {key: "undiscovered" for key in g.nodes}
    state[s] = "discovered"
    parents = {key: None for key in g.nodes}

    q = Queue()
    q.put(s)

    while not q.empty():
        v = q.get()
        # Process this vertex as desired 

        for w in g.edges[v].keys():
            # Process this edge as desired
            if state[w] == "undiscovered":
                state[w] = "discovered"
                parents[w] = v
                q.put(w)
        state[v] = "processed"

    shortest_path = [end]
    while True:
        shortest_path.append(parents[end])
        if parents[end] == s:
            break
        end = parents[end]

    shortest_path.reverse()

    exchange_product = []
    for i in range(len(shortest_path) - 1):
        curnode, nextnode = shortest_path[i], shortest_path[i + 1]
        exchange_product.append(g.edges[curnode][nextnode])
    exchange_val = reduce(mul, exchange_product)

    return state, parents, shortest_path, exchange_val


def dfs_v1a(g, s, end):
    """Outlined in Algorithms Illuminated (Tim Roughgarden)
    - g: graph
    - s: starting node
    """
    explored = {key: False for key in g.nodes}  
    explored[s] = True

    stack = deque()
    stack.append(s)

    while len(stack) > 0:
        # breakpoint()
        v = stack.pop()
        for w in g.edges[v].keys():
            print(w)
            if w == end:
                return
            if not explored[w]:
                explored[w] = True
                stack.append(w)
    return explored
