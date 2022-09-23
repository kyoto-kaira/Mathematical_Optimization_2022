#!/usr/bin/env python
# coding: utf-8

# 2段階単体法の流れ
# 1. 補助問題を単体法で解く。
# 2. 最適値が0でなければ実行不能として終了。
# 3. 最適値が0になった場合、その時の非基底変数、基底変数の割り当てに基づき、B、N、cB、cNを並び替える。
# 4. これを初期値として単体法で元の問題を解く。
# 
# 単体法本体の流れ
# 1. b_ = invB @ b と cN_ = cN - N.T @ invB.T @ cB でcN_を求める。
# 2. cN_ <= 0であれば最適解xB = b_, xN = 0, 最適値cB @ xbとして終了。
# 3. そうでなければ、cN_の正の要素ckに対応する非基底変数xkを選択。
# 4. invBNのxkに対応する列ak_がak_ <= 0であれば非有界(基底変数の非負条件を満たしながらxkを際限なく増やせる)として終了。
# 5. そうでなければ、xk = min (bi_+/aik_+) のiに対応する基底変数を選択。
# 6. 選択した非基底、基底変数およびそれらに対応するN、Bの列、cN、cBの要素を交換
# 7. 1.に戻る。

# In[ ]:


# 目的関数 max cN.T @ xN + cB.T xB
# 条件 N @ xN + B @ xB = b
#     xB >= 0
#     xN >= 0

# 等式制約からxB = invB @ b - invB @ N @ xN
# これをさっきの目的関数に代入するとcB.T @ invB @ b + cN_.T @ xN
# ここでcN_ = cN - N.T @ invB.T @ cB


# In[1]:


# https://yamagensakam.hatenablog.com/entry/2020/12/26/184808
import numpy as np
from copy import copy

# 現在どの変数が非基底変数でどの変数が基底変数にであるかについては、変数の添え字をNidx, Bidxというリストで持つことで管理
# 基底変数と非基底変数を入れ替え(BidxとNidxの中身を入れ替え)、対応するN、B、cN、cBの列も入れ替える
def swapNB(cB, cN, B, N, Bidx, Nidx, swap_pos_B, swap_pos_N):
    #idx swap 
    tmp = Nidx[swap_pos_N]
    Nidx[swap_pos_N] = Bidx[swap_pos_B]
    Bidx[swap_pos_B] = tmp

    # N, B swap
    tmp = np.copy(N[:, swap_pos_N])
    N[:, swap_pos_N] = B[:, swap_pos_B]
    B[:, swap_pos_B] = tmp

    # cN, cB, swap
    tmp = cN[swap_pos_N]
    cN[swap_pos_N] = cB[swap_pos_B]
    cB[swap_pos_B] = tmp


#単体法本体
def Simplex(cB_init, cN_init, B_init, N_init, Bidx_init, Nidx_init, b):
    Bidx = copy(Bidx_init)
    Nidx = copy(Nidx_init)
    cB = copy(cB_init)
    cN = copy(cN_init)
    B = copy(B_init)
    N = copy(N_init)
    while (1):
        # np.linalg.invで逆行列に
        invB = np.linalg.inv(B)
        b_ = invB @ b
        # N.T @ invB.T @ cB を invB.T @ cB から計算することで計算量を削減(改訂単体法)
        y = invB.T @ cB
        cN_ = cN - N.T @ y
        # 全て0以下なら停止
        if (cN_ <= 0).all() == True:
            xB = b_
            return cB @ xB, xB, cB, cN, B, N, Bidx, Nidx
        # cN_の正の要素の中で一番最初の要素に対応する変数を、交換する非基底変数として選択(最小添字規則、ブランドの規則)
        k = np.where(cN_ > 0)[0][0] # np.where() returns (np.array())

        ak_ = invB @ N[:, k]
        # 全て0以下なら非有界(xB = invB @ b - xk @ ak_ >= 0 の xkを非負制約を満たしながらいくらでも大きくできる)
        if (ak_ <= 0).all() == True:
            raise Exception('unbounded error')
        # ak_の中で、b_i/a_k_iが最小になる要素iに対応する変数を、交換する基底変数として選択
        tmp = np.ones(b_.shape)*np.inf
        tmp[ak_ > 0] = b_[ak_ > 0]/ak_[ak_ > 0]
        l = np.argmin(tmp)
        swapNB(cB, cN, B, N, Bidx, Nidx, l, k)


# bに負が含まれる場合原点が実行可能解とならない、もしくはそもそも実行可能解が存在しないことがある
# 目的 min xa
# 制約 Ax - xaI <= b
#      x >= 0
#      xa >= 0
# という補助問題をとく。xaを大きくとれば制約を満たすことはできるため実行可能解が存在する。
# 補助問題は本来の最適化問題が解を持つためにはどれくらい制約条件を修正するべきかを表し、
# 最適解がxa*=0であれば制約条件を修正する必要がなく、実行可能解が存在する。逆にxa*>=0であれば実行不能。

# 初期値を探す（2段階単体法の1段目）
def SearchFeasibleDictionary(cB_init, cN_init, B_init, N_init, Bidx_init, Nidx_init, b):
    
    bmin = np.min(b)
    if bmin < 0:
        Bidx_aux = copy(Bidx_init)
        Nidx = copy(Nidx_init)
        cB_aux = copy(cB_init)
        cN = copy(cN_init)
        B_aux = copy(B_init)
        N = copy(N_init)

        # 人工変数のindex
        artificial_var_idx = cB_init.shape[0] + cN_init.shape[0]

        # 人工変数を非基底変数にする
        Nidx_aux = copy(Nidx)
        Nidx_aux.append(artificial_var_idx)

        # 人工変数に対応する要素と列を追加
        cN_aux = np.zeros(cN.shape[0] + 1)

        # min x_a -> max -x_a　という最適化問題を解くため係数は-1
        cN_aux[cN.shape[0]] = -1
        # Nに-1の列を追加
        N_aux = np.concatenate([N, np.ones((N.shape[0], 1))*-1], axis=1)

        # bの最小値に対応する基底変数と人工変数を入れ替え
        k = np.argmin(b)
        swapNB(cB_aux, cN_aux, B_aux, N_aux, Bidx_aux, Nidx_aux, k, N_aux.shape[1] - 1)

        # 補助問題を解いて実行可能解を見つける
        z, _, res_cB, res_cN, res_B, res_N, res_Bidx, res_Nidx = Simplex(cB_aux, cN_aux, B_aux, N_aux, Bidx_aux, Nidx_aux, b)

        # 最適解が負なら実行不能(-1倍して符号を逆にしているから)
        if z < 0:
            raise Exception('infeasible error')
        # 得られた辞書のidxから人工変数を削除
        if artificial_var_idx in res_Nidx:
            # 人工変数が非基底変数にあるならそのまま削除
            res_Nidx.remove(artificial_var_idx)
        else:
            # 退化(基底変数が0になる)して最適解の基底変数に人工変数が含まれる場合は、非基底変数1つと交換する
            r = res_Bidx.index(artificial_var_idx)
            for i in range(len(res_Nidx)):
                # 非基底変数1つと交換する
                swapNB(res_cB, res_cN, res_B, res_N, res_Bidx, res_Nidx, r, i)
                if (np.linalg.matrix_rank(res_B) == res_B.shape[0]):
                    res_Nidx.remove(artificial_var_idx)
                    break
                # 正則でなければ元に戻して、交換するindexを変えてもう一度
                swapNB(res_cB, res_cN, res_B, res_N, res_Bidx, res_Nidx, i, r)

        # 得られた基底、非基底変数の割り当てに基づいて基底、非基底行列、係数を再構成
        res_cNB = np.concatenate([cN_init, cB_init])[res_Nidx + res_Bidx]
        res_cN = res_cNB[:cN_init.shape[0]]
        res_cB = res_cNB[cN_init.shape[0]:]
        res_NB = np.concatenate([N_init, B_init], axis=1)[:, res_Nidx + res_Bidx]
        res_N = res_NB[:, :cN_init.shape[0]]
        res_B = res_NB[:, cN_init.shape[0]:]
        return res_Bidx, res_Nidx, res_cB, res_cN, res_B, res_N

    # 負がなければそのままreturn
    return Bidx_init, Nidx_init, cB_init, cN_init, B_init, N_init

# max c.T @ x
# s.t. A @ x <= b
#      x >= 0
def SolveLP(c, A, b):
    cB = np.zeros(A.shape[0])
    cN = np.copy(c)
    N = np.copy(A)
    B = np.eye(A.shape[0])
    # 実際の変数のインデックス(0~n-1)
    actual_variable_idx = list(range(c.shape[0]))
    # スラック変数のインデックス(n~n+m-1)
    slack_variable_idx = list(range(c.shape[0], c.shape[0] + A.shape[0]))
    Nidx = copy(actual_variable_idx)
    Bidx = copy(slack_variable_idx)

    # 2段階単体法
    new_Bidx, new_Nidx, new_cB, new_cN, new_B, new_N = SearchFeasibleDictionary(cB, cN, B, N, Bidx, Nidx, b)
    z, xB_opt, cB_opt, cN_opt, B_opt, N_opt, Bidx_opt, Nidx_opt = Simplex(new_cB, new_cN, new_B, new_N, new_Bidx, new_Nidx, b)

    # 得られた解から実際の変数（スラック変数以外の変数）を並び替えて出力
    x_opt = np.concatenate([xB_opt, np.zeros(cN_opt.shape[0])])
    x_actual_opt = x_opt[np.argsort(Bidx_opt + Nidx_opt)][actual_variable_idx]
    return z, x_actual_opt


# In[3]:


import numpy as np

def main():
    T = int(input())
    for t in range(T):
        n, m = input().strip().split(' ')
        n, m = int(n), int(m) 
        c = np.array(list(map(float, input().strip().split(' '))))
        A = []
        for i in range(m):
            row = list(map(float, input().strip().split(' ')))
            A.append(row)
        A = np.array(A)
        b = np.array(list(map(float, input().strip().split(' '))))
        try:
            print(SolveLP(c, A, b))
        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()


# In[ ]:




