import numpy as np


def get_cutsum(i, ls, c):
    k = 0
    cutsum = 0
    while k<len(ls):
        cutsum += c[i][ls[k]]
        k += 1
    return cutsum


def donyokuhou(v, c):
    S_list = []
    T_list = []
    i = 0
    S_cutgain = 0
    T_cutgain = 0

    while i < len(v):
        S_cutgain = get_cutsum(i, T_list, c)
        T_cutgain = get_cutsum(i, S_list, c)
        if S_cutgain >= T_cutgain:
            v[i] = 1
            S_list.append(i)
        else:
            v[i] = 0
            T_list.append(i)
        i += 1
    
    return v


def main():
    N = int(input()) 
    #上から順にlistを読み込んでlistに格納していく。 
    c = [list(map(int, input().split())) for l in range(N)]
    v = [2]*N
    print(donyokuhou(v, c))


if __name__ == "__main__":
    main()
