def bellman_ford(V,E,s,inf=10**16):
    f=[inf]*V
    f[s]=0
    for _ in range(V):
        for u,v,cost in E:
            if f[v]>f[u]+cost:
                f[v]=f[u]+cost
    
    for _ in range(V):
        for u,v,cost in E:
            if f[v]>f[u]+cost:
                return None

    return f

# print(bellman_ford(3,
#          [[0, 1, 1],
#           [1, 0, 2],
#           [1, 2, 3],
#           [2, 0, 4],
#           [1, 0, 1],
#           [0, 1, 2],
#           [2, 1, 3],
#           [0, 2, 4]], 0))

print(bellman_ford(7,
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
          [6, 3, -5],
          ], 0))

# 答え[0, 3, 5, 9, 2, 8, 6]