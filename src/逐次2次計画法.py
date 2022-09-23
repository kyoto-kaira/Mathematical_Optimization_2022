import numpy as np
from 内点法 import *


def sequential_quadratic(f, gs, x0, B0, rho, gamma, d_threshold=0.01, tau=1e-4, beta=0.9):
    """
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        gs (List [function]): 制約関数のリスト
        x0 (np.ndarray): 初期点。nx1の縦ベクトル
        B0 (np.ndarray): Bの初期値。nxnの行列
        rho (float): ペナルティ関数の重みρ
        gamma (float): Bを更新する際の係数γ
        d_threshold (float): 終了するためのdのノルムの閾値
        tau (float): 直線探索に使用するアルミホ条件のτ
        beta (float): 直線探索に使用するバックトラック法の公比β
    Returns:
    """

    n = x0.shape[0]
    m = len(gs)

    # ============================== STEP1 ==============================
    x = x0
    B = B0
    k = 0
    history = [(x, B, None)]

    while True:
        # ============================== STEP2 ==============================
        # p145 式(3.207)を解く
        grad_f = grad(f, x)
        grad_gs = [grad(g, x) for g in gs]

        # 式(3.207)
        objective = lambda d: (0.5 * d.T@B@d + grad_f.T@d)[0,0]    # def(d): return (0.5 * d.T@B@d + grad_f.T@d)[0,0] と同じ
        condition = [lambda d: gs[i](x) + (grad_gs[i].T@d)[0,0] for i in range(m)]
        # 実行可能な初期値を求める
        # https://darden.hatenablog.com/entry/2016/09/05/212147
        d_init = np.zeros((n, 1))
        while True:
            c = np.array([condition[i](d_init) for i in range(m)])
            if np.max(c) <= 0:
                break
            k = np.argmax(c)
            d_init = -gs[k](x) / grad_gs[k]

        history_ipm = internal_point_method(objective, condition, m,
                                            x0=d_init,
                                            s0=np.array([-condition[i](d_init) for i in range(m)]).reshape(m, 1),
                                            u0=np.zeros((m, 1)),
                                            rho0=1.0)
        d, u_next = history_ipm[-1] # u_next=u^(k+1)

        # ============================== STEP3 ==============================
        # dのノルムが閾値未満なら終了
        if np.sqrt(np.sum(d**2)) < d_threshold:
            break

        # ============================== STEP4 ==============================
        # 直線探索でステップαを求める
        alpha = 1
        # p146 式(3.209)
        merit = lambda x: f(x) + rho * sum([max(g(x), 0) for g in gs])
        # g(α)=f(x+αd)+ρ Σ max{gi(x+αd), 0}
        g0 = merit(x)
        grad_g0 = grad(merit, x).T @ d

        # アルミホ条件を満たすまでα更新
        while merit(x+alpha*d) > g0 + tau * grad_g0 * alpha:
            alpha *= beta

        x_next = x + alpha * d  # x_next=x^(k+1)

        # ============================== STEP5 ==============================
        # p146 式(3.212)を用いてBを更新する
        # 式(3.211)
        s = x_next - x
        grad_L_next = grad(f, x_next)
        for i in range(m):
            grad_L_next += u_next[i][0] * grad(gs[i], x_next)
        
        if k == 0:  # k=0の時はu^(0)=0
            grad_L = grad(f, x)
        else:
            grad_L = history[-1][-1]

        y = grad_L_next - grad_L

        Bs = B @ s
        sBs = s.T @ Bs
        # 式(3.213)
        if s.T @ y >= gamma * sBs:
            y_bar = y
        else:
            beta = (1-gamma) * sBs / (sBs - s.T@y)
            y_bar = beta * y + (1-beta) * Bs

        # 式(3.212)
        B = B - Bs@Bs.T/sBs + y_bar@y_bar.T/(s.T@y_bar)

        x = x_next
        history.append( (x, B, grad_L_next) )
        k += 1

    return history



def f(x):
    x1 = x[0][0]
    x2 = x[1][0]
    return (x1 - 2)**4 + (x1 - 2*x2)**2

def g(x):
    x1 = x[0][0]
    x2 = x[1][0]
    return x1**2 - x2


x0 = np.array([1, 2]).reshape(-1, 1)
B0 = np.identity(2)

history = sequential_quadratic(f, [g], x0, B0, rho=10, gamma=0.2)

print("f(x~) = ", f(history[-1][0]))
print("x~ = ", history[-1][0].reshape(-1))

visualize(f, [g], history, [0,2], [0,2.5])
plt.savefig("internal.png")