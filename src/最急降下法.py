import numpy as np
import matplotlib.pyplot as plt

class SteepestDescent:
    def __init__(self, f, t1=1e-4, t2=0.5, beta=0.9, step_init=10, eps=1e-6):
        self.f = f  # 目的関数
        self.t1 = t1  # Wolfe条件のパラメータ
        self.t2 = t2
        self.beta = beta  # 公比
        self.step_init = step_init  # stepの初期値
        self.eps = eps  # 微小量
        self.x_history = []  # xの履歴

    def grad_f(self, x):
        x = np.array(x).reshape(1,-1) # np.array(1) => array(1), np.array(1).reshape(1, -1) => array([[1]])
        grad = []
        # f'(x) = (f(x+h)-f(x-h)) / 2h
        for i in range(len(x[0])):
            e = np.eye(1, x.shape[1], i) * self.eps
            grad.append((self.f(x+e) - self.f(x-e)) / (2*self.eps))
        return np.array(grad)
    
    # xの初期値, 終了する条件
    def solve(self, x_init, allow_error=1e-6):
        self.x_history.clear()
        x_now = np.array(x_init).reshape(-1,1)
        self.x_history.append(x_now.reshape(1,-1)[0])
        
        while np.sqrt((self.grad_f(x_now)**2).sum()) > allow_error:
            #x_now = self.x_history[-1].reshape(-1,1)
            d_now = - self.grad_f(x_now).reshape(-1,1)

            # ステップ幅を求める
            step = self.step_init
            while self.f(x_now + step * d_now) > self.f(x_now) + self.t1 * self.grad_f(x_now) @ d_now * step \
             or self.grad_f(x_now + step*d_now) @ d_now < self.t2 * self.grad_f(x_now) @ d_now:
                step *= self.beta
            
            # xを更新
            x_now = x_now + step * d_now
            self.x_history.append(x_now.reshape(1,-1)[0])

    def get_x(self):
        return self.x_history[-1]

# テスト１
def f(x):
    x = np.array(x).reshape(1,-1)
    x1 = x[0][0]
    x2 = x[0][1]
    return (x1 - 2)**4 + (x1 - 2*x2)**2

# テスト２
def f2(x):
    x = np.array(x).reshape(1,-1)
    x1 = x[0][0]
    x2 = x[0][1]
    return 2*x1**2 + x1*x2 + x2**2 - 5*x1 - 3*x2 + 4

steep = SteepestDescent(f, step_init=1)

steep.solve([0,3])
print(steep.get_x())

x = [p[0] for p in steep.x_history]
y = [p[1] for p in steep.x_history]
plt.plot(x,y)
plt.scatter(x,y, marker=".")
plt.scatter(x[-1],y[-1], c="red")
plt.axhline(1, c="gray", linestyle="--")
plt.axvline(2, c="gray", linestyle="--")
plt.savefig('aaa.png')
plt.show()
