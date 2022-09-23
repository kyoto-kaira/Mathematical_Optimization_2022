def quick_sort(arr):
    left = []
    right = []
    if len(arr) <= 1:
        return arr

    # データの状態に左右されないためにrandom.choice()を用いることもある。
    # ref = random.choice(arr)
    ref = arr[0]
    ref_count = 0

    for ele in arr:
        if ele < ref:
            left.append(ele)
        elif ele > ref:
            right.append(ele)
        else:
            ref_count += 1
    left = quick_sort(left)
    right = quick_sort(right)
    return left + [ref] * ref_count + right


def kruskal(V, E):
    d = quick_sort(E)
    T = []
    e = []
    label = list(range(V))
    while True:
        if len(T) == V - 1:
            break
        min_d = d.pop(0)

        if label[min_d[1]] != label[min_d[2]]:
            e.append([min_d[1], min_d[2]])
            T.append(min_d[0])
            indexes = [i for i, x in enumerate(label) if x == label[min_d[2]]]
            for index in indexes:
                label[index] = label[min_d[1]]

        else:
            continue

    return sum(T), e


# (辺の長さ、頂点、頂点)
E = [(10, 0, 1), (30, 0, 4), (10, 1, 2), (20, 1, 4),
     (30, 2, 3), (20, 4, 2), (10, 4, 3)]
print(kruskal(5, E))


# class UnionFind:
#     def __init__(self, n):
#         self.up_bound = list(range(n))
#         self.rank = [0]*n
#     def find(self, x_index):
#         if self.up_bound[x_index] == x_index:
#             return x_index
#         self.up_bound[x_index] = self.find(self.up_bound[x_index])
#         return self.up_bound[x_index]
#     def union(self, x_index, y_index):
#         repr_x = self.find(x_index)
#         repr_y = self.find(y_index)
#         if repr_x == repr_y:
#             return False
#         if self.rank[repr_x] == self.rank[repr_y]:
#             self.rank[repr_x] += 1
#             self.up_bound[repr_y] = repr_x
#         elif self.rank[repr_x] > self.rank[repr_y]:
#             self.up_bound[repr_y] = repr_x
#         else:
#             self.up_bound[repr_x] = repr_y
#         return True

# def kurskal(graph, weight):
#     u_f = UnionFind(len(graph))
#     edges = []
#     for u, _ in enumerate(graph):
#         for v in graph[u]:
#             edges.append((weight[u][v], u, v))
#     edges.sort()
#     ans = 0
#     for w_idx, u_idx, v_idx in edges:
#         if u_f.union(u_idx, v_idx):
#             ans+= w_idx
#     return ans
# N, M = map(int, input().split())
# graph = [[] for i in range(N)]
# weight = [[float('inf') for i in range(N)] for j in range(N)]
# for _ in range(M):
#     i, j, w = map(int, input().split())
#     graph[i-1].append(j-1)
#     weight[i-1][j-1] = w
# print(kurskal(graph, weight))
