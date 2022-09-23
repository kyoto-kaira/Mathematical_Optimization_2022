import numpy as np
import heapq


def prim(V, E):
    T = []
    S = []
    S.append(np.random.randint(0, V))
    while True:
        if len(S) == V:
            break
        d = []
        for i in range(len(E)):
            if E[i][1] in S and E[i][2] not in S:
                d.append(E[i])
            if E[i][2] in S and E[i][1] not in S:
                d.append(E[i])

        min_e = min(d)

        T.append(min_e[0])

        if min_e[1] in S:
            S.append(min_e[2])
        else:
            S.append(min_e[1])

    return sum(T)


def prim2(V, E, inf=10**16):
    # vを始点とするi番目の辺について
    # g[v][i][0]は終点となる頂点, g[v][i][1]は辺の長さ
    # を表す
    # 無向グラフは辺2本の有向グラフに直す
    g = [[] for _ in range(V)]

    used = [False]*V
    for e in E:
        g[e[1]].append([e[2], e[0]])
        g[e[2]].append([e[1], e[0]])

    # 頂点0から出る辺を列挙
    que = []
    used[0] = True
    for nv, cost in g[0]:
        heapq.heappush(que, [cost, nv])

    s = 0
    while len(que) > 0:
        cost, v = heapq.heappop(que)
        if used[v]:
            continue

        # Sのカットのうち、辺の長さが最小のもの:頂点v_iのi番目の辺
        used[v] = True
        s += cost

        for nv, cost in g[v]:
            # 新しい頂点を加えたら、その頂点を端点とする辺(Sのカットの辺)を取ってくる。
            if not used[nv]:
                heapq.heappush(que, [cost, nv])
    return s


E = [(10, 0, 1), (30, 0, 4), (10, 1, 2), (20, 1, 4),
     (30, 2, 3), (20, 4, 2), (10, 4, 3)]

print(prim2(5, E))
