import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import norm

sys.path.append('../sampledata')
import generate_data as gen


def generateData(n):
    # データ生成関数
    # n:生成するデータの個数
    # np.reshape(X, (-1, 1)) : 縦ベクトル生成
    gen_noize = gen.GenerateRandomNoize(mean=0, sd=0.3, n=n)
    gen_noize.generate()
    noize = np.reshape(gen_noize.value, (-1, 1))

    uniform = gen.GenerateUniform(low=0, high=1.0, n=n)
    uniform.generate()

    x = np.reshape(uniform.value, (-1, 1))
    y = np.sin(2*np.pi*x) + noize
    return x, y


class BayesCurveFittingModel:
    def __init__(self, x, y, M):
        self.x = x
        self.y = y
        self.M = M
        self.I = np.eye(M+1)

    def calcE(self):
        # 正則化・ペナルティ項を持たない場合
        # Aの計算
        self.A = np.zeros((self.M+1, self.M+1))
        for i in range(self.M+1):
            for j in range(self.M+1):
                self.A[i, j] = np.sum(self.x**(i+j))

        # Tの計算
        self.T = np.zeros(self.M+1)
        for i in range(self.M+1):
            self.T[i] = np.dot((self.x**i).T, self.y)

        # 方程式Aw=Tの計算によりwを求める
        self.w = np.linalg.solve(self.A, self.T)
        return self.w

    def calcRegularE(self, lamb):
        # 正則化・ペナルティ項を持つ場合
        # lamb=0で正則化項を持たない場合と等価
        # Aの計算
        self.A = np.zeros((self.M+1, self.M+1))
        for i in range(self.M+1):
            for j in range(self.M+1):
                self.A[i, j] = np.sum(self.x**(i+j))

        # Tの計算
        self.T = np.reshape(np.zeros(self.M+1), (-1, 1))
        for i in range(self.M+1):
            self.T[i] = np.dot((self.x**i).T, self.y)

        # \lamb*Iの計算
        self.lambI = self.I * lamb

        # 方程式(A+\lamb*I)w=Tの計算によりwを求める
        self.w = np.linalg.solve(self.A+self.lambI, self.T)
        return self.w

    def predictModel(self, x_pred):
        # (1.1)式の計算
        # xの生成・整形
        self.x_predict_matrix = x_pred**0
        for j in range(1, self.M+1):
            self.x_predict_matrix = np.hstack([np.c_[self.x_predict_matrix], np.c_[x_pred**j]])

        # モデルを用いた予測
        self.y_predict = np.dot(self.x_predict_matrix, self.w)

        return self.y_predict

    def phi(self, x):
        # ベクトルphi
        return np.matrix(np.array([x**i for i in range(M+1)]))

    def predictDist(self, alpha, beta, x_predict):
        # 予測分布
        # 行列Sの計算処理
        self.dotphi = np.zeros((self.M+1, self.M+1))
        for n in range(len(self.x)):
            self.dotphi += np.dot(self.phi(self.x[n]), self.phi(self.x[n]).T)

        S_inv = alpha*self.I + beta*self.dotphi
        m = np.array([])
        s = np.array([])
        for n in range(len(x_predict)):
            # 予測分布の平均m(x)
            tmp1 = np.dot(self.phi(x_predict[n]).T, np.linalg.inv(S_inv))
            tmp2 = np.reshape(np.array([np.dot(self.y.T, self.phi(self.x)[m, :].T) for m in range(self.M+1)]), (-1, 1))
            tmp3 = beta * np.dot(tmp1, tmp2)
            m = np.append(m, tmp3)

            # 予測分布の分散s^2(x)
            tmp4 = beta**(-1) + np.dot(np.dot(self.phi(x_predict[n]).T, np.linalg.inv(S_inv)), self.phi(x_predict[n]))
            s = np.append(s, tmp4)

        return m, s


if __name__ == '__main__':
    # データ生成
    x_data, y_data = generateData(n=10)
    x_predict = np.reshape(np.arange(0, 1.0, 0.01), (-1, 1))

    for M in [0, 1, 3, 5, 9, 16]:

        # モデル作成
        model = BayesCurveFittingModel(x_data, y_data, M)

        # wの計算
        # 正則化項なし
        # model.calcE()

        # 正則化項あり、lambdaはexpをつけないと大きすぎる
        model.calcRegularE(lamb=np.exp(-18))

        # 予測分布(プラスマイナス標準偏差)
        # プロットについて、参考にさせていただきました
        # http://qiita.com/Gordian_knot/items/555802600638f41b40c5
        m, s = model.predictDist(alpha=5*10**(-3), beta=11.1, x_predict=x_predict)
        y_pred_h = m + np.sqrt(s)
        y_pred_m = m
        y_pred_l = m - np.sqrt(s)
        plt.fill_between(np.reshape(x_predict, (-1)), y_pred_h, y_pred_l, color='pink')

        # 学習データ
        y_predict = model.predictModel(x_predict)
        plt.scatter(x_data, y_data)

        # 予測分布の平均線
        plt.plot(x_predict, y_pred_m, 'r')
        # 予測した曲線
        # plt.plot(x_predict, y_predict, 'r')
        # 正解の曲線
        plt.plot(x_predict, np.sin(2*np.pi*x_predict), 'g')

        plt.xlim((0, 1.0))
        plt.ylim((-2.0, 2.0))
        plt.title('Predictive Distribution M = %d' % M)
        plt.show()
