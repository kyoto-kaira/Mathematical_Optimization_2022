def greedy(V, E, c):
    """
    Args:
        V (int): 頂点数
        E (List[List[int]]): 辺のリスト
        c (List[int]): 各頂点のコスト
    Returns:
        S (List[bool]): 各頂点が選択されたかどうかのリスト
    """

    # ========================= STEP1 ============================
    S = [False for _ in range(V)]
    nums_E_uncovered = [0 for _ in range(V)] # 頂点vから出る、被覆されていない辺の数
    num_E_covered = 0 # 被覆された辺の数
    score = 0  # 選択された頂点のコスト和
    G = [[] for _ in range(V)]

    for v, u in E:
        G[v].append(u)
        G[u].append(v)
        nums_E_uncovered[u] += 1
        nums_E_uncovered[v] += 1

    # ========================= STEP2 ============================
    while num_E_covered < len(E):
        # ========================= STEP3 ============================
        # 費用効果が最小になるv*を求める
        m = 100000000000
        v_min = 0
        for v in range(V):
            # すでに選ばれた頂点は飛ばす
            if S[v]:
                continue

            # vから出る辺が全て被覆されてれば選ぶ必要はない
            if nums_E_uncovered[v] == 0:
                continue

            cost_eff = c[v] / nums_E_uncovered[v]
            if m > cost_eff:
                v_min = v
                m = cost_eff

        S[v_min] = True
        score += c[v_min]
        num_E_covered += nums_E_uncovered[v_min]
        nums_E_uncovered[v_min] = 0

        for u in G[v_min]:
            if nums_E_uncovered[u] > 0: #頂点uが選ばれていないなら
                nums_E_uncovered[u] -= 1

    return S, score


def dual(V, E, c):
    """
    Args:
        V (int): 頂点数
        E (List[List[int]]): 辺のリスト
        c (List[int]): 各頂点のコスト
    Returns:
        S (List[bool]): 各頂点が選択されたかどうかのリスト
    """

    # ========================= STEP1 ============================
    S = [False for _ in range(V)]
    c_bar = [c[v] for v in range(V)]
    y = [0 for _ in E]
    score = 0

    # ========================= STEP2 ============================
    for e, (u, v) in enumerate(E):
        # ========================= STEP3 ============================
        if S[v] or S[u]:
            continue

        # c_bar[u] >= c_bar[v]となるよう(u,v)を決める
        if c_bar[u] <= c_bar[v]:
            u, v = v, u

        y[e] += c_bar[v]
        c_bar[u] -= c_bar[v]
        c_bar[v] = 0
        S[v] = True
        score += c[v]

    return S, score


print(dual(
    4,
    [[0,1], [1,2], [0,2], [1,3], [2,3]],
    [3, 4, 5, 3]
))

# 最適解は[1,2]が選択される時
# 貪欲法だと近似比率が 1+1/2+1/3+1/4+1/5 = 2.28
# 主双対法だと近似比率が 2



"""
# test
import random

V = random.randint(10, 1000)
E = []
c = []
for u in range(V):
    c.append( random.randint(1, 100) )
    for v in range(u+1, V):
        if random.random() < 0.5:
            E.append( (u, v) )

print(f"V = {V}")
print(f"|E| = {len(E)}")
print(f"貪欲法の近似比率: {sum([1/(i+1) for i in range(len(E))])}")
print(f"主双対法の近似比率: 2")


def check(S):
    for u, v in E:
        if not (S[u] or S[v]):
            return False

    return True


S, score = greedy(V, E, c)
print(f"貪欲法: {score}   valid: {check(S)}")

S, score = dual(V, E, c)
print(f"主双対法: {score}   valid: {check(S)}")
"""