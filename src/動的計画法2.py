import numpy as np

def dp(C, p, w):
    n = len(w)
    P = sum(p)
    # 品物のindex, 価値の合計->重さの最小値
    g = np.zeros((n, P + 1))
    x = np.zeros((P + 1, n))

    # Step1：g(1,v)を計算する
    for v in range(P + 1):
        if v == 0:
            g[0][v] = 0
        elif v == p[0]:
            g[0][v] = w[0]
            x[v][0] = 1
        else:
            g[0][v] = np.inf

    # Step2：k=n ならば終了、そうでなければ k+=1
    k = 1
    while True:
        if k == n:
            break
        # Step3：g(k,v)を計算
        x_ = x.copy()
        for v in range(P + 1):
            if v < p[k]:
                g[k][v] = g[k-1][v]
            else:
                g_ = [g[k-1][v], g[k-1][v-p[k]] + w[k]]
                g[k][v] = min(g_)
                if np.argmin(g_) == 1:
                    x[v] = x_[v-p[k]]
                    x[v][k] = 1
        
        k += 1
    
    for v in range(P + 1):
        if g[n-1][v] <= C:
            ans = v

    return ans, x[ans]

#　本の例1
p = [3, 4, 1, 2]
w = [2, 3, 1, 3]
C = 4

print(dp(C, p, w))