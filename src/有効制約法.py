import numpy as np
import matplotlib.pyplot as plt

class ActiveSet:
    def __init__(self, Q, c, A, b, eps=1e-6):
        self.Q = Q  # 目的関数
        self.c = c
        self.A = A  # 制約関数
        self.b = b
        self.eps = eps
        self.dim = Q.shape[0]  # 変数xの次元
        self.I = []  # 有効制約条件の添え字集合
        self.x_history = []  # ｘの履歴

    # 目的関数
    def f(self, x):
        x = np.array(x).reshape(-1,1)
        y = 0.5 * x.T @ self.Q @ x + self.c.T @ x
        return y[0][0]

    # 目的関数の勾配
    def grad_f(self, x):
        x = np.array(x).reshape(-1,1)
        y = self.Q @ x + self.c
        return y

    def solve(self, x_init, allow_error=1e-6):
        x_now = np.array(x_init).reshape(-1,1)
        assert all(self.A @ x_now >= self.b), "初期解が実行不可能です"

        self.x_history.clear()
        self.x_history.append(x_now.reshape(1,-1)[0])
        
        flg = True
        while True:
            # Step2
            if flg:
                self.I.clear()
                is_active = abs(self.A @ x_now - self.b) < self.eps
                for i,cond in enumerate(is_active):
                    if cond:
                        self.I.append(i)
            
            # Step3
            # 係数行列
            m = len(self.I)
            B = np.zeros((self.dim + m, self.dim+m))
            bb = np.concatenate([-self.c, self.b[self.I]])

            for i in range(self.dim):  # 1つ目の等式の変数xの係数を格納
                for j in range(self.dim):
                    B[i][j] = self.Q[i][j]
            for i in range(self.dim):  # 1つめの等式の変数uの係数を格納
                for j in range(m):
                    B[i][self.dim+j] = - self.A[self.I].T[i][j]
            for i in range(m):  # 2つ目の等式の変数xの係数を格納
                for j in range(self.dim):
                    B[self.dim+i][j] = self.A[self.I][i][j]

            solution = np.linalg.solve(B, bb).reshape(-1,1)
            x_bar = solution[:self.dim]
            u = solution[self.dim:]

            # Step4
            if any(abs(x_bar - x_now) > self.eps):
                flg = True
                if all(self.A @ x_bar >= self.b):
                    x_now = x_bar
                    self.x_history.append(x_now.reshape(1,-1)[0])
                    continue
                else:
                    alphas = []
                    for i in range(self.A.shape[0]):
                        if self.A[i] @ (x_bar - x_now) != 0:
                            alpha = - (self.A[i] @ x_now - self.b[i]) / (self.A[i] @ (x_bar - x_now))
                            if 0 < alpha[0] <= 1:
                                alphas.append(alpha[0])
                    print(alphas)
                    alpha = min(alphas)
                    x_now = x_now + alpha * (x_bar - x_now)
                    self.x_history.append(x_now.reshape(1,-1)[0])
                    continue
            
            # Step5
            if all(u >= 0):
                break

            # Step6
            i_at_u = np.argmin(u)
            i = self.I[i_at_u]
            self.I.remove(i)
            self.x_history.append(x_now.reshape(1,-1)[0])
            flg = False

    def get_x(self):
        return self.x_history[-1]



Q = np.array([[8,-4],
              [-4,6]])
c = np.array([-8,0]).reshape(-1,1)

A = np.array([[-1,-1]])

b = np.array([-4]).reshape(-1,1)

t = ActiveSet(Q,c,A,b)
t.solve([2,2])
print(t.get_x())
