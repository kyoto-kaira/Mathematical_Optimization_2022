import heapq
from multiprocessing import heap


def dijkstra(V, E, s, t, inf=10**16):
    # vを始点とするi番目の辺について
    # g[v][i][0]は終点となる頂点, g[v][i][1]は辺の長さ
    # を表す
    # 無向グラフは辺2本の有向グラフに直す
    g = [[] for _ in range(V)]

    used = [False]*V
    for e in E:
        g[e[0]].append([e[1], e[2]])
        # g[e[1]].append([e[0], e[2]])

    # 頂点sから出る辺を列挙
    que = []
    f = [inf]*V
    f[s] = 0
    # 最短距離、始点
    heapq.heappush(que, [0, s])
    while len(que) > 0:
        cost, v = heapq.heappop(que)
        if used[v] or cost != f[v]:
            continue

        # Sのカットのうち、辺の長さが最小のもの:頂点v_iのi番目の辺
        used[v] = True

        for nv, cost in g[v]:
            # 新しい頂点を加えたら、その頂点を端点とする辺(Sのカットの辺)を取ってくる。
            if not used[nv] and f[nv] > f[v]+cost:
                f[nv] = f[v]+cost
                heapq.heappush(que, [cost, nv])
    return f[t]


# print(dijkstra(3,
#          [[0, 1, 1],
#           [1, 0, 2],
#           [1, 2, 3],
#           [2, 0, 4]], 0, 2))

print(dijkstra(7,
         [[0, 1, 4],
          [0, 4, 2],
          [1, 2, 2],
          [1, 4, 2],
          [1, 5, 6],
          [2, 3, 4],
          [2, 5, 3],
          [2, 6, 1],
          [4, 1, 1],
          [4, 2, 4],
          [5, 4, 1],
          [5, 3, 2],
          [5, 6, 2],
          [6, 3, 5],
          ], 0, 6))