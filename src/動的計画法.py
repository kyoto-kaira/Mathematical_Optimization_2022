import numpy as np

def dp(C, w, p):
    n = len(w)
    # 品物のindex, 袋の容量->価値の最大値
    f = np.zeros((n, C + 1))
    # 選んだ荷物
    x = np.zeros((C + 1, n))

    # Step1：f(1,u)を計算する
    for u in range(C + 1):
        if u < w[0]:
            f[0][u] = 0
        else:
            f[0][u] = p[0]
            x[u][0] = 1
    
    # Step2：k=n ならば終了、そうでなければ k+=1
    k = 1
    while True:
        if k == n:
            break
        # Step3：f(k,u)を計算
        x_ = x.copy()
        for u in range(C + 1):
            if u < w[k]:
                f[k][u] = f[k-1][u]
            else:
                f_ = [f[k-1][u], f[k-1][u-w[k]] + p[k]]
                f[k][u] = max(f_)
                if np.argmax(f_) == 1:
                    x[u] = x_[u-w[k]]
                    x[u][k] = 1
        
        k += 1
    
    return f[n-1][u], x[C]

#　本の例
C = 7
w = [2, 3, 4, 5]
p = [16, 19, 23, 28]

C = 8
w = [3, 4, 5]
p = [30, 50, 60]

C = 5
w = [1, 1, 1, 1, 1]
p = [1000000000, 1000000000, 1000000000, 1000000000, 1000000000, 1000000000]

C = 15
w = [6, 5, 6, 6, 3, 7]
p = [5, 6, 4, 6, 5, 2]

print(dp(C, w, p))
            