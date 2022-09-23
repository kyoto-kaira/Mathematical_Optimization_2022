import numpy as np

def dp(F, B):
    """
    F : 関数のリスト
    B : 総資源量
    """
    n = len(F)
    # 関数のindex, 資源の量->利益の最大値
    f = np.zeros((n, B+1))
    x = np.zeros((B+1, n))
    
    # Step1: f(1,u) = f_1(u) (u=0,...,B)とする
    for u in range(B+1):
        f[0][u] = F[0](u)
        x[u][0] = u
    k = 1
    
    # step2: k=nなら終了、そうでなければk+=1 
    while True:
        if k == n:
            break
        x_ = x.copy()
        # Step3: f(k,u) (u=0,...,B)を計算
        for u in range(B + 1):
            d = []
            for x_k in range(u+1):
                d.append(f[k-1][u-x_k] + F[k](x_k))
            f[k][u] = max(d)
            x_k = np.argmax(d)
            x[u] = x_[u-x_k]
            x[u][k] = x_k
        
        k += 1

    return f[n-1][B], x[B]

def f_1(x):
    return 10 * abs(x - 1)

def f_2(x):
    return x ** 2 

def f_3(x):
    return 2 * (x - 1) ** 2

def f_4(x):
    return 60 / abs(2 * x -9)

F = [f_1, f_2, f_3, f_4]

print(dp(F, 6))