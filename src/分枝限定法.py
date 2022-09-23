
import numpy as np
import simplex_method
import copy
import math
import sys
import warnings
warnings.filterwarnings('ignore')

"""
Args:
    配列の確認
        set_problem: 線形緩和問題を格納している三次元配列
        P: 線形緩和問題。二次元配列。
        x_now: 線形緩和問題の最適解 
        z_now: 線形緩和問題の最適値
        x_ops: 整数計画問題の暫定最適解
        z_ops: 整数計画問題の暫定最適値
Returns:
"""

# c[1*n], A[n*m],b[m*1] を線形緩和問題として保存
# 保存先は,set_problem


def simplex_method_main(ploblem_info):
    # 入力ではなく、問題を受け取るパターンの単体法
    c = copy.copy(ploblem_info[0])
    A = copy.copy(ploblem_info[1])
    b = copy.copy(ploblem_info[2])

    try:
        return simplex_method.SolveLP(np.array(c), np.array(A), np.array(b))
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

# 解が全て整数かを判定し、整数計画問題の実行可能解フラッグと、非整数のindexを返す。


def check_int(x):
    for i in range(len(x)):
        if int(x[i]) == x[i]:
            continue
        else:
            return False, i  # Falseとindex
    return True, None  # Trueとindex


# problem:c, A, bの順番に入っている
def make_small_problems(problem, x, nonint_index):
    b_new1 = []
    b_new2 = []
    A_new1 = []
    A_new2 = []
    a = x[nonint_index]
    b_new1 = problem[2] + [math.floor(a)]
    b_new2 = problem[2] + [-1*math.ceil(a)]

    new_row_A = [0]*len(problem[0])
    for i in range(len(problem[0])):
        if i == nonint_index:
            new_row_A[i] = 1
        else:
            new_row_A[i] = 0
    A_new1 = np.vstack([problem[1], new_row_A]).tolist()
    
    for j in range(len(problem[0])):
        if j == nonint_index:
            new_row_A[j] = -1
        else:
            new_row_A[j] = 0

    A_new2 = np.vstack([problem[1], new_row_A]).tolist()

    new_problem1 = np.copy(problem)
    new_problem1[1] = A_new1
    new_problem1[2] = b_new1
    new_problem2 = np.copy(problem)
    new_problem2[1] = A_new2
    new_problem2[2] = b_new2

    return new_problem1, new_problem2


def main():
    set_problem = []  # c, A, bを入れた配列
    # 整数計画問題をP0をセット (set_problem[0] = [P0])
    set_problem.append(set_first_problem())
    z_ops = -1 * np.inf  # 暫定解 -∞ を暫定解としてセット
    x_ops = [0]*len(set_problem[0][1])
    while set_problem:
        current_prob = set_problem.pop()
        # print(np.array(current_prob[0]))
        # print(np.array(current_prob[1]))
        # print(np.array(current_prob[2]))
        simp_ans = simplex_method_main(current_prob)
        # print(f"本家:{simp_ans}")
        if not isinstance(simp_ans, tuple):
            if str(simp_ans) == 'unbounded error':
                print('unsolvable error')
                return
            elif str(simp_ans) == 'infeasible error':
                continue
            else:
                print(simp_ans)
                return
        else:
            z_now = simp_ans[0]
            x_now = simp_ans[1]
            # 緩和問題の最適解が整数計画問題の実行可能解(整数)ならば暫定解を更新
            judge_int = check_int(x_now)
            if judge_int[0]:  # 整数計画問題の実行可能解か(整数か)をチェック
                if z_ops < z_now:  # 暫定値と比較。大きければ更新。
                    z_ops = z_now
                    x_ops = x_now
                    continue

            # 最適解が整数でない場合は、問題Pに分枝操作を適用して、set_problemに追加してループの先頭に戻る
            else:  # 子問題を作成してset_problemに格納
                new_problem1, new_problem2 = make_small_problems(
                    current_prob, x_now, judge_int[1])
                set_problem.append(new_problem1)
                set_problem.append(new_problem2)
    return z_ops, x_ops  # 最終的な暫定値と暫定解を出力


if __name__ == '__main__':
    print(main())


"""
4 5
3 4 1 2
2 3 1 3
1 0 0 0
0 1 0 0
0 0 1 0 
0 0 0 1 
4 1 1 1 1
"""