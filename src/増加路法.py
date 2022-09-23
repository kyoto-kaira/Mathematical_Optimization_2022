class edge:
    def __init__(self, to,cap,rev):
        self.to=to
        self.cap=cap
        self.rev=rev

def build_graph(V,E):
    g=[[] for _ in range(V)]
    for u,v,cap in E:
        g[u].append(edge(v,cap,len(g[v])))
        g[v].append(edge(u,0,len(g[u])-1))
    return g

def ford_fulkerson(V,E,s,t,inf=10**16):
    g=build_graph(V,E)
    #vが現在の頂点, fがvから流れるフローの最大値
    def dfs(v,f):
        if v==t:
            return f
        used[v]=True

        for e in g[v]:
            if used[e.to] or e.cap<=0:
                continue

            d=dfs(e.to,min(f,e.cap))
            if d>0:
                e.cap-=d
                g[e.to][e.rev].cap+=d
                return d
        return 0
        
    flow = 0
    while True:
        used=[False]*V
        f=dfs(s,inf)
        if f==0:
            return flow
        flow+=f

print(ford_fulkerson(
    6,
    [
        [0,1,10],
        [0,3,40],
        [1,3,10],
        [1,2,20],
        [2,5,40],
        [3,2,15],
        [3,4,20],
        [4,2,15],
        [4,5,20]
    ],
    0,5
))