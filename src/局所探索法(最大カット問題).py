import random
import numpy as np
import copy


def get_cutsum(i, ls, c):
    k = 0
    cutsum = 0
    while k < len(ls):
        cutsum += c[i][ls[k]]
        k += 1
    return cutsum


def local_search_method(v, c):
    S_list = []
    T_list = []

    # # Sにいくつかの頂点を所属させ、それ以外の頂点をTに所属させる。
    # r = random.randint(1, len(v))
    # for k in range(r):
    #     S_list.append(k)  # Sの所属リストにindexを追加していく
    #     v[k] = 1  # 所属フラグを立てる
    # for k in range(r, len(v)):
    #     T_list.append(k)
    
    for k in range(len(v)):
        idx=random.randint(0,1)
        v[k]=idx
        if idx==1:
            S_list.append(k)
        else:
            T_list.append(k)

    i = 0
    not_use_idx = -1
    while i < len(v):
        if i == not_use_idx:
            i += 1
            continue

        # v[i]がSに属していて、「V[i]をSから出した時のカット」＞「v[i]をSにそのまま所属させた時のカット」 の場合v[i]をSの所属から外す(所属リストから除外する)。
        if v[i] == 1 and get_cutsum(i, list(set(S_list) - set([i])), c) - get_cutsum(i, T_list, c) > 0:
            S_list.remove(i)
            T_list.append(i)
            v[i] = 0
            not_use_idx = i
            i = 0
        # v[i]がSに属していなくて、「Sに入れた時のカット」＞「Sから出したままの時のカット」の場合v[i]をSへ所属させる(所属リストに追加する)
        elif v[i] == 0 and get_cutsum(i, list(set(T_list) - set([i])), c) - get_cutsum(i, S_list, c) > 0:
            S_list.append(i)
            T_list.remove(i)
            v[i] = 1
            not_use_idx = i
            i = 0
        else:
            i += 1
    return S_list


def main():
    # tableを受け取る
    N = int(input())
    # 上から順にlistを読み込んでlistに格納していく。
    c = [list(map(int, input().split())) for l in range(N)]
    # 各頂点の所属を全て0(L)にする
    v = [0]*N
    print(local_search_method(v, c))


if __name__ == "__main__":
    main()
