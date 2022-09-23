import numpy as np
from functools import partial
from optim import *
import matplotlib.pyplot as plt

class AugmentedLagrange:
    def __init__(self, f, g_eq=[], beta=2, eps=1e-6):
        self.f = f  # 目的関数
        self.g_eq = g_eq  # 等式制約のリスト
        self.beta = beta  #　公比
        self.eps = eps  # 十分小さい値
        self.x_history = [] # xの履歴

    # ペナルティ関数
    def penalty(self, x, rho):
        return rho * np.sum([g(x)**2 for g in self.g_eq])

    # ラグランジュ関数
    def lagrange(self, x, u):
        return self.f(x) + np.sum([u_ *g(x) for u_,g in zip(u, self.g_eq)])

    # 拡張ラグランジュ関数
    def aug_lagrange(self, x, u, rho):
        return self.lagrange(x,u) + self.penalty(x, rho)

    def solve(self, x_init, u_init, rho_init, opt):
        x_now = np.array(x_init).reshape(-1,1)
        u_now = np.array(u_init).reshape(-1,1)
        rho_now = rho_init
        self.x_history.clear()
        self.x_history.append(x_now.reshape(1,-1)[0])

        while self.penalty(x_now, rho_now) > self.eps:
            # Step2
            optimizer = opt(partial(self.aug_lagrange, u=u_now, rho=rho_now))
            optimizer.solve(x_init=x_now)
            x_now = optimizer.get_x()
            self.x_history.append(x_now.reshape(1,-1)[0])

            # Step4 uを更新
            u_now = u_now + rho_now * np.array([g(x_now) for g in self.g_eq]).reshape(-1,1)
            rho_now *= self.beta

    def get_x(self):
        return self.x_history[-1] 

# テスト1
def f1(x):
    x = np.array(x).reshape(-1,1)
    x1 = x[0][0]
    x2 = x[1][0]
    return (x1 - 2)**4 + (x1 - 2*x2)**2

def g1(x):
    x = np.array(x).reshape(-1,1)
    x1 = x[0][0]
    x2 = x[1][0]
    return x1**2 - x2


t = AugmentedLagrange(f=f1, g_eq=[g1])
t.solve(x_init=[2,1], u_init=[0], rho_init=1, opt=QuasiNewton)

print(t.get_x())
X = t.x_history
x = [p[0] for p in X]
y = [p[1] for p in X]
plt.plot(x,y)
plt.scatter(x,y)
plt.scatter(0.945582993415968406929394915, 0.894127197437503351608508082, c="r")
plt.savefig("fig1.png")
plt.show()

# 解(Wolframalpha)
# 0.945582993415968406929394915, 0.894127197437503351608508082