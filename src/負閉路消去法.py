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
def negative_path_deletion(V, E, b, inf=10**16):
    g = build_graph(V, E)

    g.append([])
    sp = V
    V += 1
    res = 0
    for v in range(V-1):
        res += abs(b[v])*inf
        if b[v] > 0:
            g[v].append(edge(sp, inf-b[v], inf, len(g[sp])))
            g[sp].append(edge(v, b[v], -inf, len(g[v])-1))
        if b[v] < 0:
            g[sp].append(edge(v, inf-(-b[v]), inf, len(g[v])))
            g[v].append(edge(sp, -b[v], -inf, len(g[sp])-1))

    while True:
        f = [inf]*V
        f[sp] = 0
        prev = [[] for _ in range(V)]
        for _ in range(V):
            for v in range(V):
                for i, e in enumerate(g[v]):
                    if e.cap > 0 and f[e.to] > f[v]+e.cost:
                        f[e.to] = f[v]+e.cost
                        prev[e.to] = [v, i]

        idx = []
        update = False
        for v in range(V):
            if update:
                break
            for i, e in enumerate(g[v]):
                if e.cap > 0 and f[e.to] > f[v]+e.cost:
                    used = [False]*V
                    used[e.to] = True
                    p = [v, i]
                    while not used[p[0]]:
                        used[p[0]] = True
                        p = prev[p[0]]
                    idx.append(p)
                    pe = p
                    p = prev[p[0]]
                    while p[0] != pe[0]:
                        idx.append(p)
                        p = prev[p[0]]
                    update = True
                    break

        if not update:
            break

        delta = inf
        for v, e in idx:
            delta = min(delta, g[v][e].cap)
            v = g[v][e].to

        for v, e in idx:
            res += g[v][e].cost*delta
            g[v][e].cap -= delta
            g[g[v][e].to][g[v][e].rev].cap += delta
            v = g[v][e].to

    return res


print(negative_path_deletion(
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
print(negative_path_deletion(
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

print(negative_path_deletion(
    2,
    [[0, 1, 1, -4]],
    [1,-1]
))

print(negative_path_deletion(
    3,
    [
        [0, 1, 2, -4],
        [1, 2, 2, -4],
        # [0, 2, 1, -100]
    ],
    [2, 0, -2]
))
