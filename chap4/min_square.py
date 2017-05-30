# coding:utf-8
# 最小二乗誤差による線形分類

import sys
import numpy as np
import random as rd
import matplotlib.pyplot as plt

sys.path.append('../sampledata')
import generate_data as gen


class Data:
    def __init__(self, mean, cov, N, cls, maxcls):
        self.N = N
        # 1-of-K 符号化によってクラスを表すベクトルを生成
        self.clsvec = np.zeros(maxcls)
        self.clsvec[cls-1] = 1
        # 2次元正規分布に従ってデータ生成(ヘンにクラスを使いすぎた)
        gauss2d = gen.GenerateRandomNoize2D(mean=mean, cov=cov, n=N)
        gauss2d.generate()
        # 生成した座標を格納
        self.x1 = np.reshape(gauss2d.x, (len(gauss2d.x), 1))
        self.x2 = np.reshape(gauss2d.y, (len(gauss2d.y), 1))
        self.x_vec = np.hstack((self.x1, self.x2))

    def scatter(self, color, marker='o'):
        plt.scatter(x=self.x1, y=self.x2, color=color, marker=marker, alpha=0.4)


def calcX(x):
    '''
    * X~の計算
    * X~の擬似逆行列の計算
    '''
    # X~ の作成
    dummy = np.ones((len(x), 1))
    X_tilde = np.hstack((dummy, x))
    # print(data.X_tilde)

    # X~のムーアペンローズの擬似逆行列(pseudo-inverse matrix) X_pimの計算
    # (X^T * X)^-1
    tmp = np.linalg.inv(np.dot(X_tilde.T, X_tilde))
    # (X^T * X)^-1 * X^T
    X_pim = np.dot(tmp, X_tilde.T)

    # 擬似逆行列を一発で求める方法
    # X_pim = np.linalg.pinv(data.X_tilde)

    return X_tilde, X_pim


def calcT(clsvec, N):
    # T の作成
    T = np.array([clsvec for i in range(N)])
    # print(T)

    return T


def calcY(T, X_pim, x_input):
    # 識別関数の計算(4.17)式
    y = np.dot(np.dot(T.T, X_pim.T), x_input.T)


def calcBoundary(T, X_pim, x_input, cls_k, cls_j):
    # クラスkとクラスjの境界面を計算する
    # ここやべえ
    cls_k = cls_k - 1
    cls_j = cls_j - 1
    x_input = np.arange(-30, 30, 0.1)
    w = np.dot(X_pim, T)
    w0 = w[0, cls_k] - w[0, cls_j]
    w1 = (w[1:, cls_k]-w[1:, cls_j])[0]
    w2 = (w[1:, cls_k]-w[1:, cls_j])[1]

    x2 = (-w0 - w1*x_input) / w2
    plt.plot(x_input, x2, color='black')
    # print(x2)


def __main():
    # クラス数
    maxcls = 2
    # データ数
    N = 50
    # 各クラスのデータの平均・分散
    mean_1 = np.array([0, 0])
    mean_2 = np.array([0, 5.5])
    cov = np.array([[1, -0.7], [-0.7, 1]])
    # Dataクラスを用いてデータをN個生成
    data_1 = Data(mean=mean_1, cov=cov, N=N, cls=1, maxcls=maxcls)
    data_2 = Data(mean=mean_2, cov=cov, N=N, cls=2, maxcls=maxcls)

    # クラス2に属する外れ値を意図的に生成
    data_out = Data(mean=np.array([10, 10]), cov=cov, N=10, cls=2, maxcls=maxcls)

    # 各クラスのデータ点を一つの行列にまとめる(外れ値なし)
    # x_vec = np.vstack((data_1.x_vec, data_2.x_vec))
    # 各クラスのデータ点を一つの行列にまとめる(外れ値あり)
    x_vec = np.vstack((data_1.x_vec, data_2.x_vec, data_out.x_vec))

    # 擬似逆行列の計算
    X_tilde, X_pim = calcX(x_vec)

    # Tの計算
    data_1.T = calcT(data_1.clsvec, data_1.N)
    data_2.T = calcT(data_2.clsvec, data_2.N)
    data_out.T = calcT(data_out.clsvec, data_out.N)

    # 各データ点のTをマージ(外れ値なし)
    # T = np.vstack((data_1.T, data_2.T))
    # 各データ点のTをマージ(外れ値なし)
    T = np.vstack((data_1.T, data_2.T, data_out.T))

    calcY(T=T, X_pim=X_pim, x_input=X_tilde)

    calcBoundary(T=T, X_pim=X_pim, x_input=0, cls_k=1, cls_j=2)

    # データ点プロット
    data_1.scatter('red', 'x')
    data_2.scatter('blue', 's')
    data_out.scatter('blue', 's')
    plt.xlim((-5, 15))
    plt.ylim((-5, 15))
    plt.title('Linear Classification (Min-Square Error)')
    plt.show()

if __name__ == '__main__':
    __main()