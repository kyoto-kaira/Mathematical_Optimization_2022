import numpy as np


def nearest_addition_algorithm(V, E):
    S = []
    d_list = []
    has_edge = [[False for _ in range(len(V))] for _ in range(len(V))]
    #ランダムに都市を決めて集合Sに入れる or 初期値を0にしておく
    r = np.random.randint(0, len(V))
    S.append(r)
    r2 = np.argmin([[E[r][i], i] for i in range(len(V)) if i != r])
    if r2 >= r:
        r2 += 1
    S.append(r2)
    has_edge[r][r2] = True
    has_edge[r2][r] = True
    d_list.append(E[r][r2])
    d_list.append(E[r2][r])
    while True:
        if len(S) == len(V):
            break
        d_min_judge = []  # 最小値を判別する用の配列を初期化
        for i in range(len(V)):
            if i in S:
                for j in range(len(V)):  # ←ここのサイズ調整可能か？Sの中まで探してる。
                    if i != j and j not in S:
                        d_min_judge.append([E[i][j], i, j])

        # minの経路の距離をGet
        min_e = min(d_min_judge)[0]
        ##minだった時のjを保存したい##
        i = min(d_min_judge)[1]
        j = min(d_min_judge)[2]
        d_list.append(min_e)
        has_edge[i][j] = True
        # Hのうちのiを含む[i,k]を消して、[i,j],[j,k]に変える
        for k in range(len(V)):
            if k != i and k != j and has_edge[i][k]:
                d_list.remove(E[i][k])
                has_edge[i][k] = False  # 古い道を消去
                # d_list.append(E[i][j]) すでに入っている
                d_list.append(E[j][k])
                # has_edge[i][j] = True  # 新しい道に更新
                has_edge[j][k] = True
                break

        S.append(j)  # minだった時のjを代入

    # print(has_edge)
    return sum(d_list)


def main():
    # V: 都市名、文字列のリスト
    V = input().split()
    distance_table = [list(map(int, input().split())) for i in range(len(V))]
    print(nearest_addition_algorithm(V, distance_table))


if __name__ == "__main__":
    main()

# a b c d
# 0 3 1 4
# 1 0 5 9
# 2 6 0 5
# 3 5 8 0
