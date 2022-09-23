import matplotlib.pyplot as plt
import numpy as np
import functools


def grad(f, x, delta=1e-6):
    """
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        x (np.ndarray): nx1の縦ベクトル
        delta (float): 微小値hの大きさ
    Returns:
        grad_f (np.ndarray): nx1の縦ベクトル∇f
    """
    n = x.shape[0]
    grad_f = np.zeros((n, 1))
    for i in range(n):
        h = np.eye(1, n, i).T * delta
        grad_f[i][0] = (f(x+h) - f(x-h)) / (2*delta)
    return grad_f


# https://research.miidas.jp/2019/06/pythonでやる多次元ニュートン法/
def hessian(f, x, delta=1e-6):
    """
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        x (np.ndarray): nx1の縦ベクトル
        delta (float): 微小値hの大きさ
    Returns:
        hessian_f (np.ndarray): nxnのヘッセ行列∇^2f
    """
    n = x.shape[0]
    hessian_f = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            # f_xy(x,y) = (f_x(x,y+h) - f_x(x,y-h)) / 2h
            # f_x(x,y+h) = (f(x+h,y+h) - f(x-h,y+h)) / 2h
            # f_x(x,y-h) = (f(x+h,y-h) - f(x-h,y-h)) / 2h
            hx = np.eye(1, n, i).T * delta
            hy = np.eye(1, n, j).T * delta

            fp = (f(x+hx+hy) - f(x-hx+hy)) / (2*delta)
            fm = (f(x+hx-hy) - f(x-hx-hy)) / (2*delta)
            hessian_f[i][j] = (fp - fm) / (2*delta)
            hessian_f[j][i] = hessian_f[i][j]
    return hessian_f


def merit_func(f, gs, l, x, s, rho, eta):
    """
    式(3.194)

    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        gs (List [function]): 制約関数のリスト。最初のl個は不等式制約で、l個目以降は等式制約
        l (int): 不等式制約の数
        x (np.ndarray): nx1の縦ベクトル
        s (np.ndarray): lx1の縦ベクトル
        rho (float): スラック変数に対するバリア関数の重みρ
        eta (float): 直線探索する際のメリット関数の重みη
    Returns:
        v (float): メリット関数の値
    """
    v = f(x)
    for i in range(l):
        v -= rho * np.log(max(s[i][0], 1e-8))
        v += eta * np.abs(gs[i](x) + s[i][0])

    for i in range(l, len(gs)):
        v += eta * np.abs(gs[i](x))

    return v


def internal_point_method(f, gs, l, x0, s0, u0, rho0, delta=0.1, eta=0.1, rho_threshold=1e-9, tau=1e-4, beta=0.9):
    """
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        gs (List [function]): 制約関数のリスト。最初のl個は不等式制約で、l個目以降は等式制約
        l (int): 不等式制約の数
        x0 (np.ndarray): 初期点。nx1の縦ベクトル
        s0 (np.ndarray): スラック変数の初期値で、非負。lx1の縦ベクトル
        u0 (np.ndarray): ラグランジュ乗数の初期値で、最初のl個は非負。nx1の縦ベクトル
        rho0 (float): スラック変数に対するバリア関数の重みρの初期値
        delta (float): ρを更新する際の係数δ
        eta (float): 直線探索する際のメリット関数の重みη
        rho_threshold (float): 終了するためのρの閾値
        tau (float): 直線探索に使用するアルミホ条件のτ
        beta (float): 直線探索に使用するバックトラック法の公比β
    Returns:
    """
    n = x0.shape[0]
    m = len(gs)

    # ============================== STEP1 ==============================
    x = x0
    s = s0
    u = u0
    rho = rho0
    history = [(x, u)]

    # ============================== STEP2 ==============================
    while rho > rho_threshold:
        # ============================== STEP3 ==============================
        # P140 式(3.193)を解く
        grad_f = grad(f, x)
        hessian_f = hessian(f, x)
        grad_gs = [grad(g, x) for g in gs]
        hessian_gs = [hessian(g, x) for g in gs]

        A = np.zeros((n+l+m, n+l+m)) # 係数行列
        c = np.zeros((n+l+m, 1))
        #     Δx
        #  A  Δs  = c
        #     Δu
        # Aの横方向は、0:nがΔx, n:n+lがΔs, n+l:n+l+mがΔuに対応する係数

        # (3.193)第1式
        A[:n, :n] = hessian_f
        c[:n] = -grad_f
        for i in range(m):
            A[:n, :n] += u[i][0] * hessian_gs[i]
            A[:n, n+l+i] = grad_gs[i][:,0]
            c[:n] -= u[i][0] * grad_gs[i]

        # (3.193)第2式
        for i in range(l):
            A[n+i, n+i] = u[i,0]
            A[n+i, n+l+i] = s[i,0]
            c[n+i] = rho - u[i,0] * s[i,0]

        # (3.193)第3式
        for i in range(l):
            A[n+l+i, :n] = grad_gs[i][:,0]
            A[n+l+i, n+i] = 1
            c[n+l+i] = -gs[i](x) - s[i,0]
        
        # (3.193)第4式
        for i in range(l, m):
            A[n+l+i, :n] = grad_gs[i][:,0]
            c[n+l+i] = -gs[i](x)
        
        # 連立方程式を解いてΔx, Δs, Δuを求める
        xsu = np.linalg.solve(A, c)
        dx = xsu[:n]
        ds = xsu[n:n+l]
        du = xsu[n+l:]

        # 直線探索でステップαを求める
        alpha = 1
        # アルミホ条件の g(α)=f(x+αd)
        # fはメリット関数、dは(Δx, Δs)^T
        g0 = merit_func(f, gs, l, x, s, rho, eta)
        merit_fixed_s = functools.partial(merit_func, f, gs, l, s=s, rho=rho, eta=eta)   # xのみを引き受ける
        merit_fixed_x = functools.partial(merit_func, f, gs, l, x, rho=rho, eta=eta)     # sのみを引き受ける
        # P99 式(3.75)
        grad_g0 = np.vstack([grad(merit_fixed_s, x), grad(merit_fixed_x, s)]).T @ np.vstack([dx, ds])


        # 更新後のスラック変数sとラグランジュ乗数uは負になってはいけない
        while not (np.all(s + alpha*ds >= 0) and np.all(u + alpha*du >= 0)):
            alpha *= beta
        
        # アルミホで条件満たすまでαを更新
        while merit_func(f, gs, l, x+alpha*dx, s+alpha*ds, rho, eta) > g0 + tau * grad_g0*alpha:
            alpha *= beta

        # 更新先が最小となるαを求める
        # alpha_prev = alpha
        # while merit_func(f, gs, l, x+alpha*dx, s+alpha*ds, rho, eta) <= merit_func(f, gs, l, x+alpha_prev*dx, s+alpha_prev*ds, rho, eta):
        #     alpha_prev = alpha
        #     alpha *= beta

        # ============================== STEP4 ==============================
        x = x + alpha * dx
        u = u + alpha * du
        s = s + alpha * ds
        rho = delta * u.T @ s / l   # ρの更新(p141, 式3.195)

        print(alpha, rho, dx.reshape(-1), ds.reshape(-1), du.reshape(-1))
        history.append( (x,u) )

    return history



def visualize(f, gs, history, x_lim, y_lim, N=100, step=50, alpha=0.3):
    x_pts = np.linspace(x_lim[0], x_lim[1], N)
    y_pts = np.linspace(y_lim[0], y_lim[1], N)
    x_grid, y_grid = np.meshgrid(x_pts, y_pts)
    data = np.stack([x_grid, y_grid], -1).reshape(-1, 1, 2).transpose(2,1,0)
    z = f(data).reshape(N, N)

    contf = plt.contourf(x_grid, y_grid, z, step, cmap="PuOr", alpha=alpha)
    plt.colorbar(contf)

    for g in gs:
        g_value = g(data).reshape(N, N)
        g_is0_y = y_pts[np.argmin(np.abs(g_value), axis=0)]
        plt.plot(x_pts, g_is0_y, linestyle="--")

    x_history = np.array([his[0].reshape(-1) for his in history])
    plt.plot(x_history[:,0], x_history[:,1], marker="o")
    plt.plot(x_history[0,0], x_history[0,1], marker="o", color="black")
    plt.plot(x_history[-1,0], x_history[-1,1], marker="o", color="red")



# =============================================================
# p142 実行例
# =============================================================
def f(x):
    x1 = x[0][0]
    x2 = x[1][0]
    return (x1-2)**4 + (x1-2*x2)**2

def g(x):
    x1 = x[0][0]
    x2 = x[1][0]
    return x1**2 - x2


if __name__ == "__main__":
    x0 = np.array([1, 2]).reshape(-1, 1)
    s0 = np.array([1]).reshape(-1, 1)
    u0 = np.array([0]).reshape(-1, 1)

    history = internal_point_method(f, [g], l=1, x0=x0, s0=s0, u0=u0, rho0=1, beta=0.9)

    print("f(x~) = ", f(history[-1][0]))
    print("x~ = ", history[-1][0].reshape(-1))

    visualize(f, [g], history, [0,2], [0,2.5])
    plt.savefig("internal.png")