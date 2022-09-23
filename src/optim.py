import numpy as np

##### 最急降下法 #####
class SteepestDescent:
    def __init__(self, f, t1=1e-4, t2=0.5, beta=0.9, step_init=10, eps=1e-6):
        self.f = f  # 目的関数
        self.t1 = t1  # Armijo条件のパラメータ
        self.t2 = t2  # 曲率条件のパラメータ
        self.beta = beta  # バックトラック法のパラメータ（公比）
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
        
        iter = 1
        while np.sqrt((self.grad_f(x_now)**2).sum()) > allow_error:
            d_now = - self.grad_f(x_now).reshape(-1,1)

            # ステップ幅を求める
            step = self.step_init
            while self.f(x_now + step * d_now) > self.f(x_now) + self.t1 * self.grad_f(x_now) @ d_now * step \
             or self.grad_f(x_now + step*d_now) @ d_now < self.t2 * self.grad_f(x_now) @ d_now:
                step *= self.beta
                if step < 1e-16:
                    print("Warning: line search was early stopped (iter={iter})")
                    break
            
            # xを更新
            x_now = x_now + step * d_now
            self.x_history.append(x_now.reshape(1,-1)[0])
            iter += 1

    def get_x(self):
        return self.x_history[-1]

##### Newton法 #####
class Newton:
    def __init__(self, f, h, t1=1e-4, t2=0.5, beta=0.9, step_init=10, eps=1e-6, line_search=False):
        self.f = f  # 目的関数
        self.h = h  # 目的関数のヘッセ行列
        self.t1 = t1  # Armijo条件のパラメータ
        self.t2 = t2  # 曲率条件のパラメータ
        self.beta = beta  # バックトラック法のパラメータ（公比）
        self.step_init = step_init  # stepの初期値
        self.eps = eps  # 微小量
        self.line_search = line_search  # 直線探索するかどうか
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
        
        iter = 1
        while np.sqrt((self.grad_f(x_now)**2).sum()) > allow_error:
            d_now = - np.linalg.inv(self.h(x_now)) @ self.grad_f(x_now).reshape(-1,1)

            # ステップ幅を求める
            if self.line_search:
                step = self.step_init
                while self.f(x_now + step * d_now) > self.f(x_now) + self.t1 * self.grad_f(x_now) @ d_now * step \
                or self.grad_f(x_now + step*d_now) @ d_now < self.t2 * self.grad_f(x_now) @ d_now:
                    step *= self.beta
                    if step < 1e-16:
                        print("Warning: line search was early stopped (iter={iter})")
                        break
            
            # xを更新
            if not self.line_search:
                step = 1
            x_now = x_now + step * d_now
            self.x_history.append(x_now.reshape(1,-1)[0])
            iter += 1

    def get_x(self):
        return self.x_history[-1]


##### 準Newton法 #####
class QuasiNewton:
    def __init__(self, f, t1=1e-4, t2=0.5, beta=0.9, step_init=10, eps=1e-6):
        self.f = f  # 目的関数
        self.t1 = t1  # Armijo条件のパラメータ
        self.t2 = t2  # 曲率条件のパラメータ
        self.beta = beta  # バックトラック法のパラメータ（公比）
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
        B_now = np.eye(len(x_init))
        self.x_history.append(x_now.reshape(1,-1)[0])
        
        n_iter = 1
        while np.sqrt((self.grad_f(x_now)**2).sum()) > allow_error:
            d_now = - np.linalg.inv(B_now) @ self.grad_f(x_now).reshape(-1,1)

            # ステップ幅を求める
            flg = False
            step = self.step_init
            while self.f(x_now + step * d_now) > self.f(x_now) + self.t1 * self.grad_f(x_now) @ d_now * step \
            or self.grad_f(x_now + step*d_now) @ d_now < self.t2 * self.grad_f(x_now) @ d_now:
                step *= self.beta
                if step < 1e-9:
                    print(f"Warning: line search was early stopped (iter={n_iter})")
                    flg = True
                    break
            if flg:
                break
            
            # xを更新
            x_now = x_now + step * d_now
            self.x_history.append(x_now.reshape(1,-1)[0])

            # Bを更新
            x_before = self.x_history[-2].reshape(-1,1)
            s = x_now - x_before
            y = (self.grad_f(x_now) - self.grad_f(x_before)).reshape(-1,1)
            B_now = B_now - np.dot(B_now@s, (B_now@s).T) / (s.T@B_now@s)[0][0] + (y@y.T) / (s.T@y)[0][0]
            n_iter += 1

    def get_x(self):
        return self.x_history[-1]

##### 有効制約法 #####
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
        assert all(self.A @ x_now >= self.b), "initial solution is infeasible."

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