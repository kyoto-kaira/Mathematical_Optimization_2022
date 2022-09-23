# https://algo-method.com/tasks/316

import numpy as np

def matching(c):
    """
    c : 各ペアをマッチさせたときのコスト
    """
    m = len(c)
    n = len(c[0])
    f = np.zeros((m, n))
    # Step1
    f[0][0] = c[0][0]
    # f(i, 1)をもとめる
    for i in range(1, m):
        f[i][0] = f[i-1][0] + c[i][0]
    j = 1
    # Step2
    while True:
        if j == n:
            break
        # Step3
        # f(1, j)を求める
        f[0][j] = f[0][j-1] + c[0][j]
        # f(i, j)を求める
        for i in range(1, m):
            f_ = [f[i-1][j-1], f[i-1][j], f[i][j-1]]
            f[i][j] = min(f_) + c[i][j]
        
        j += 1

    return f[m-1][n-1]

c = [[1, 2, 3], [4, 3, 2]]
print(matching(c))