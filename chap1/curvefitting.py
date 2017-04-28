import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import norm

sys.path.append('../sampledata')
import generate_data as gen


def generateData(n):
    # データ生成関数
    # n:生成するデータの個数
    gen_noize = gen.GenerateRandomNoize(mean=0, sd=0.3, n=n)
    gen_noize.generate()
    noize = gen_noize.value

    uniform = gen.GenerateUniform(low=0, high=1.0, n=n)
    uniform.generate()

    x = uniform.value
    y = (np.sin(2*np.pi*x)+noize)
    return x, y


class CurveFittingModel:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def calcE(self, M):
        # 正則化・ペナルティ項を持たない場合
        # Aの計算
        self.A = np.zeros((M+1, M+1))
        for i in range(M+1):
            for j in range(M+1):
                self.A[i, j] = np.sum(self.x**(i+j))

        # Tの計算
        self.T = np.zeros(M+1)
        for i in range(M+1):
            self.T[i] = np.dot(self.x**i, self.y)

        # 方程式Aw=Tの計算によりwを求める
        self.w = np.linalg.solve(self.A, self.T)
        return self.w

    def calcRegularE(self, M, lamb):
        # 正則化・ペナルティ項を持つ場合
        # lamb=0で正則化項を持たない場合と等価
        # Aの計算
        self.A = np.zeros((M+1, M+1))
        for i in range(M+1):
            for j in range(M+1):
                self.A[i, j] = np.sum(self.x**(i+j))

        # Tの計算
        self.T = np.zeros(M+1)
        for i in range(M+1):
            self.T[i] = np.dot(self.x**i, self.y)

        # \lamb*Iの計算
        self.I = np.eye(M+1)
        self.lambI = self.I * lamb

        # 方程式(A+\lamb*I)w=Tの計算によりwを求める
        self.w = np.linalg.solve(self.A+self.lambI, self.T)
        return self.w

    def predictModel(self, M, x_pred):
        # (1.1)式の計算
        # xの生成・整形
        self.x_predict_matrix = np.array([x_pred**j for j in range(M+1)]).T

        # モデルを用いた予測
        self.y_predict = np.dot(self.x_predict_matrix, self.w)

        return self.y_predict


if __name__ == '__main__':
    # データ生成
    x_data, y_data = generateData(n=10)
    model = CurveFittingModel(x_data, y_data)

    x_predict = np.arange(0, 1.0, 0.01)

    for M in [0, 1, 3, 6, 9]:
        # wの計算
        # 正則化項なし
        # model.calcE(M)

        # 正則化項あり、lambdaはexpをつけないと大きすぎる
        model.calcRegularE(M=M, lamb=np.exp(-18))
        # model.calcRegularE(M=M, lamb=0)

        y_predict = model.predictModel(M, x_predict)
        plt.scatter(x_data, y_data)

        # 予測した曲線
        plt.plot(x_predict, y_predict, 'r')
        # 正解の曲線
        plt.plot(x_predict, np.sin(2*np.pi*x_predict), 'g')

        plt.xlim((0, 1.0))
        plt.ylim((-2.0, 2.0))
        plt.title('Curve Fitting M = %d' % M)
        plt.show()
