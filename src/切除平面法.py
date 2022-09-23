import simplex_method
import numpy as np
import math
import copy as cp
import sys


def simplex_method_main(problem_info):
    # 入力ではなく、問題を受け取るパターンの単体法
    c = cp.copy(problem_info[0])
    A = cp.copy(problem_info[1])
    b = cp.copy(problem_info[2])

    try:
        return *simplex_method.SolveLP2(np.array(c), np.array(A), np.array(b)), problem_info
    except Exception as e:
        tb = sys.exc_info()[2]
        return e.with_traceback(tb)


def set_first_problem():
    first_problem = []
    n, m = map(int, input().split(' '))
    c = list(map(float, input().strip().split(' ')))
    A = []
    for i in range(m):
        row = list(map(float, input().strip().split(' ')))
        A.append(row)
    b = list(map(float, input().strip().split(' ')))
    first_problem = [c, A, b]
    return first_problem


def check_int(x):
    for i in range(len(x)):
        if int(x[i]) == x[i]:
            continue
        else:
            return False, i  # Falseとindex
    return True, None  # Trueとindex


# problem:c, A, bの順番に入っている
# x_opt=b_bar
def update_problem(c, A, b, A_bar, N_idx, B_idx, x_opt, nonint_index):
    b_new = []
    A_new = []

    # 新しい制約条件を計算
    # index= kのところにfloor「\bar{a}i」をつけて、xiに「1」をつける
    # 右辺には、floor \bar{b}iを入れる

    # p255式4.126
    # \bar{a}_iと\bar{b}_iの定義
    # \bar{b}=B^{-1}b
    # 最適値x=(x_B, x_N)に対してx_B=B^{-1}bなので\bar{b}=x(=x_opt)
    # \bar{A}=B^{-1}N なのでBとNが必要

    b_new = b + [math.floor(x_opt[nonint_index])]

    new_row_A = [0]*len(c)
    new_row_A[nonint_index] = 1
    # \bar{a}_ik
    # ただし\bar{A} = B^{-1}Nで、\bar{b}=x=B'{-1}b

    for k, i in enumerate(N_idx):
        if i < len(c):
            new_row_A[i] += cp.copy(math.floor(
                A_bar[B_idx.index(nonint_index)][k]))
    A_new = np.vstack([A, new_row_A]).tolist()

    new_problem = [None, None, None]
    new_problem[0] = cp.copy(c)
    new_problem[1] = A_new
    new_problem[2] = b_new

    return new_problem


def main():
    # 問題の入力
    problem = set_first_problem()
    # 単体法で線形緩和問題の最適解をGet
    z, x_opt, Nidx, Bidx, A_bar, problem = simplex_method_main(problem)
    # 整数チェック→辞書を更新(切る)→最適解→整数チェック→辞書を更新(切る)→...
    while True:
        print("最適解：")
        print(x_opt)
        print(problem[1])
        print(problem[2])

        if check_int(x_opt)[0]:  # 線形緩和問題の解が整数計画問題の実行可能解ならば終了
            return z, x_opt

        nonint_index = check_int(x_opt)[1]

        z, x_opt, Nidx, Bidx, A_bar, problem = cp.copy(simplex_method_main(update_problem(
            problem[0], problem[1], problem[2], A_bar, Nidx, Bidx, x_opt, nonint_index)))


if __name__ == '__main__':
    print(main())

# 制約は等式制約を想定していることに注意


""" 
4 2
0 1 0 0
2 3 1 0
-2 1 0 1
6 0
"""

#いけないケース
# 2 2
# 0 1
# 2 3 
# -2 1
# 6 0