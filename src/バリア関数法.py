import numpy as np
from functools import partial
from optim import *
from ペナルティ関数法 import Penalty 

class Barrier:
    # ペナルティ関数はnが1=>| |, 2=>()^2
    def __init__(self, f, g_neq=[], kind="1/x", beta=0.5, eps=-1e+8):
        self.f = f  #目的関数
        self.g_neq = g_neq # 不等式制約のリスト
        self.kind = kind  # バリア関数の種類（'1/x' or 'log'）
        self.beta = beta # rhoの公比
        self.eps = eps  # 微小量
        self.x_history = []   # ｘの履歴

    def penalty(self, x, rho):
        if self.kind == "1/x":
            p_neq = -np.sum([1/g(x) for g in self.g_neq])
        if self.kind == "log":
            p_neq = -np.sum([np.log(-g(x)) for g in self.g_neq])
        return rho * p_neq
    
    def f_p(self, x, rho):
        return self.f(x) + self.penalty(x, rho)

    def solve(self, x_init, rho_init, opt):
        x_now = np.array(x_init).reshape(-1,1)
        rho_now = rho_init
        self.x_history.clear()
        self.x_history.append(x_now.reshape(1,-1)[0])

        while self.penalty(x_now, rho_now) > self.eps:
            optimizer = opt(partial(self.f_p, rho=rho_now), step_init=1)
            optimizer.solve(x_init=x_now)
            x_now = optimizer.get_x()
            self.x_history.append(x_now.reshape(1,-1)[0])

            rho_now *= self.beta
    
    def get_x(self):
        return self.x_history[-1]

# テスト1
def f1(x):
    x = np.array(x).reshape(-1,1)
    return (0.1*(x-3.5)**2)[0][0]

def g1_1(x):
    x = np.array(x).reshape(-1,1)
    return (x-3)[0][0]

def g1_2(x):
    x = np.array(x).reshape(-1,1)
    return (0.5-x)[0][0]

t = Barrier(f=f1, g_neq=[g1_1, g1_2], kind="log", beta=0.9, eps=-1e+8)
t.solve(x_init=[2], rho_init=0.2, opt=QuasiNewton)

print(t.get_x())
print(t.x_history)

