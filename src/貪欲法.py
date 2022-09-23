# https://yamaimo.hatenablog.jp/entry/2020/03/14/200000
import numpy as np

def greedy(B, apple):
    n = len(apple)  # 配分先の数
    x = [0] * n
    d = np.zeros(n)
    while True:
        if sum(x) == B:
            break
        for j in range(n):
            d[j] = apple[j][x[j] + 1] - apple[j][x[j]]
        max_d = np.argmax(d)
        x[max_d] += 1
    
    return x

apple = [[0, 13, 24, 33, 42, 48 ,52], 
         [0, 17, 29, 37, 42, 45, 46], 
         [0, 16, 31, 38, 41, 43, 44], 
         [0, 20, 38, 52, 62, 67, 69]]

print(greedy(6, apple))