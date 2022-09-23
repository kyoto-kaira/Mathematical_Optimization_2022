def floyd_warshall(V,E,inf=10**16):
    d=[[inf for _ in range(V)] for _ in range(V)]
    for v in range(V):
        d[v][v]=0

    for u,v,cost in E:
        d[u][v]=cost

    for k in range(V):
        for u in range(V):
            for v in range(V):
                d[u][v] = min(d[u][v], d[u][k]+d[k][v])
    
    for v in range(V):
        if d[v][v]<0:
            return None

    return d


print(floyd_warshall(5,
[
    [0,1,3],
    [0,4,-4],
    [0,2,8],
    [1,3,1],
    [1,4,7],
    [2,1,4],
    [3,0,2],
    [3,2,-5],
    [4,3,6]
]
))