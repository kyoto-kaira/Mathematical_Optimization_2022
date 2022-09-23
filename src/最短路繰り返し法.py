import heapq


class edge:
    def __init__(self, to, cap, cost, rev):
        self.to = to
        self.cap = cap
        self.cost = cost
        self.rev = rev


def build_graph(V, E):
    g = [[] for _ in range(V)]
    for u, v, cap, cost in E:
        g[u].append(edge(v, cap, cost, len(g[v])))
        g[v].append(edge(u, 0, -cost, len(g[u])-1))
    return g

# 辺[from, to,  cap, cost]


def shortest_path(V, E, b, inf=10**16):
    g = build_graph(V, E)

    g.append([])
    g.append([])
    s = V
    t = V+1
    V += 2
    f = 0
    res = 0
    for v in range(V-2):
        for e in g[v]:
            if e.cap > 0 and e.cost < 0:
                c = e.cap
                res += e.cost*c
                g[e.to][e.rev].cap += c
                e.cap = 0
                b[v] -= c
                b[e.to] += c

    for v in range(V-2):
        if b[v] > 0:
            f += b[v]
            g[s].append(edge(v, b[v], 0, len(g[v])))
            g[v].append(edge(s, 0, 0, len(g[s])-1))
        if b[v] < 0:
            g[v].append(edge(t, -b[v], 0, len(g[t])))
            g[t].append(edge(v, 0, 0, len(g[v])-1))

    while f > 0:
        dist = [inf]*V
        dist[s] = 0
        prev = [[] for _ in range(V)]
        for _ in range(V-1):
            for v in range(V):
                if dist[v] == inf:
                    continue
                for i, e in enumerate(g[v]):
                    if e.cap > 0 and dist[e.to] > dist[v]+e.cost:
                        dist[e.to] = dist[v]+e.cost
                        prev[e.to] = [v, i]

        if dist[v] == inf:
            return None

        d = f
        v = t
        while v != s:
            d = min(d, g[prev[v][0]][prev[v][1]].cap)
            v = prev[v][0]

        f -= d
        res += d*dist[t]
        v = t
        while v != s:
            g[prev[v][0]][prev[v][1]].cap -= d
            g[v][g[prev[v][0]][prev[v][1]].rev].cap += d
            v = prev[v][0]
    return res


def shortest_path2(V, E, b, inf=10**16):
    g = build_graph(V, E)

    g.append([])
    g.append([])
    s = V
    t = V+1
    V += 2
    f = 0
    res = 0
    for v in range(V-2):
        for e in g[v]:
            if e.cap > 0 and e.cost < 0:
                c = e.cap
                res += e.cost*c
                g[e.to][e.rev].cap += c
                e.cap = 0
                b[v] -= c
                b[e.to] += c

    for v in range(V-2):
        if b[v] > 0:
            f += b[v]
            g[s].append(edge(v, b[v], 0, len(g[v])))
            g[v].append(edge(s, 0, 0, len(g[s])-1))
        if b[v] < 0:
            g[v].append(edge(t, -b[v], 0, len(g[t])))
            g[t].append(edge(v, 0, 0, len(g[v])-1))

    h = [0]*V
    while f > 0:
        dist = [inf]*V
        prev = [[] for _ in range(V)]
        que = []
        dist[s] = 0
        heapq.heappush(que, [0, s])
        while len(que) > 0:
            d, v = heapq.heappop(que)
            if dist[v] < d:
                continue

            for i, e in enumerate(g[v]):
                if e.cap > 0 and dist[e.to] > dist[v]+e.cost+h[v]-h[e.to]:
                    dist[e.to] = dist[v]+e.cost+h[v]-h[e.to]
                    prev[e.to] = [v, i]
                    heapq.heappush(que, [dist[e.to], e.to])

        if dist[t] == inf:
            return None

        for v in range(V):
            h[v] += dist[v]

        d = f
        v = t
        while v != s:
            d = min(d, g[prev[v][0]][prev[v][1]].cap)
            v = prev[v][0]

        f -= d
        res += d*h[t]
        v = t
        while v != s:
            g[prev[v][0]][prev[v][1]].cap -= d
            g[v][g[prev[v][0]][prev[v][1]].rev].cap += d
            v = prev[v][0]
    return res


print(shortest_path2(
    5,
    [
        [0, 1, 20, 3],
        [0, 2, 40, 7],
        [2, 1, 10, 2],
        [2, 3, 50, 3],
        [3, 1, 20, 4],
        [3, 4, 30, 8],
        [1, 4, 40, 3],
    ],
    [50, 0, 0, 0, -50]
))

# https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_6_B
print(shortest_path2(
    4,
    [
        [0, 1, 2, 1],
        [0, 2, 1, 2],
        [1, 2, 1, 1],
        [1, 3, 1, 3],
        [2, 3, 2, 1]
    ],
    [2, 0, 0, -2]
))

print(shortest_path2(
    3,
    [
        [0, 1, 2, -4],
        [1, 2, 2, -4],
        [0, 2, 1, -100]
    ],
    [2, 0, -2]
))
