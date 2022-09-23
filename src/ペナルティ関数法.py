import numpy as np
from functools import partial
from optim import * 

class Penalty:
    # ペナルティ関数はnが1=>| |, 2=>()^2
    def __init__(self, f, g_eq=[], g_neq=[], n=1, beta=2, eps=1e-6):
        self.f = f  #目的関数
        self.g_eq = g_eq # 等式制約のリスト
        self.g_neq = g_neq # 不等式制約のリスト
        self.n = n # ペナルティ関数の次数
        self.beta = beta # rhoの公比
        self.eps = eps  # 微小量
        self.x_history = []   # ｘの履歴

    def penalty(self, x, rho):
        p_neq = np.sum([max(g(x),0)**self.n for g in self.g_neq])
        p_eq = np.sum([abs(g(x))**self.n for g in self.g_eq])
        return rho * ( p_neq + p_eq)
    
    def f_p(self, x, rho):
        return self.f(x) + self.penalty(x, rho)

    def solve(self, x_init, rho_init, opt):
        x_now = np.array(x_init).reshape(-1,1)
        rho_now = rho_init
        self.x_history.clear()
        self.x_history.append(x_now.reshape(1,-1)[0])

        while self.penalty(x_now, rho_now) > self.eps:
            optimizer = opt(partial(self.f_p, rho=rho_now))
            optimizer.solve(x_init=x_now)
            x_now = optimizer.get_x()
            self.x_history.append(x_now.reshape(1,-1)[0])

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

t = Penalty(f=f1, g_eq=[g1], n=2)
t.solve(x_init=[2,1], rho_init=1, opt=QuasiNewton)

print(t.get_x())

