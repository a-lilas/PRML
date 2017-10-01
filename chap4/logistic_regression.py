# coding:utf-8
# ロジスティック回帰, ニュートン・ラフソン法

import sys
import numpy as np
import random as rd
import matplotlib.pyplot as plt

sys.path.append('../sampledata')
import generate_data as gen


class Data:
    def __init__(self, mean, cov, N, cls, maxcls):
        self.N = int(N)
        self.cls = cls
        # 2次元正規分布に従ってデータ生成(ヘンにクラスを使いすぎた)
        gauss2d = gen.GenerateRandomNoize2D(mean=mean, cov=cov, n=N)
        gauss2d.generate()
        # 生成した座標を格納
        self.x1 = np.reshape(gauss2d.x, (len(gauss2d.x), 1))
        self.x2 = np.reshape(gauss2d.y, (len(gauss2d.y), 1))
        self.x_vec = np.hstack((self.x1, self.x2))

    def scatter(self, color, marker='o', ax=False):
        if ax:
            ax.scatter(x=self.x1, y=self.x2, color=color, marker=marker, alpha=0.4)
        else:
            plt.scatter(x=self.x1, y=self.x2, color=color, marker=marker, alpha=0.4)


def sigmoid(a):
    # logistic sigmoid function (4.59)
    return 1 / (1 + np.exp(-a))


def convertFeatureVector(x):
    # @input    x: data(2d-vector)
    # @output   phi: non-linear converted feature (vector 2*1)
    phi = np.array([[1, x[0], x[1]]])
    return phi


def plotW(w, ax=False):
    # separation plane: w1*x1 + w2*x2 + w0 = 0
    x = np.arange(-30, 30, 0.1)
    y = -(w[0] + w[1]*x) / w[2]

    if ax:
        ax.plot(x, y, color='green')
    else:
        plt.plot(x, y, color='green')


def __main():
    # データ生成
    # クラス数
    maxcls = 2
    # 総データ数
    N = 200
    # 各クラスのデータの平均・分散
    mean_1 = np.array([0, 0])
    mean_2 = np.array([0, 8])
    cov = np.array([[10, -1.6], [-1.6, 10]])
    # Dataクラスを用いてデータをN個生成
    data_1 = Data(mean=mean_1, cov=cov, N=int(N/2), cls=0, maxcls=maxcls)
    data_2 = Data(mean=mean_2, cov=cov, N=int(N/2), cls=1, maxcls=maxcls)
    # クラス2に属する外れ値を意図的に生成
    # out_N = 30
    # N += out_N
    # data_out = Data(mean=np.array([20, 20]), cov=cov, N=out_N, cls=1, maxcls=maxcls)
    # data_2.x_vec = np.vstack((data_2.x_vec, data_out.x_vec))

    # feature degree
    feature_d = 3
    # initialize weights (shape: maxcls * 1)
    w = np.zeros((feature_d, 1))

    # non-linear convert
    data_1.phi_vec = np.empty((0, feature_d))
    data_2.phi_vec = np.empty((0, feature_d))
    for i, data in enumerate(data_1.x_vec):
        data_1.phi_vec = np.append(data_1.phi_vec, convertFeatureVector(data), axis=0)
    for i, data in enumerate(data_2.x_vec):
        data_2.phi_vec = np.append(data_2.phi_vec, convertFeatureVector(data), axis=0)

    # design matrix
    design_phi = np.vstack((data_1.phi_vec, data_2.phi_vec))

    # calc class vector (t)
    # TODO: refactoring
    t_vec = np.empty((0, 1))
    for _ in range(data_1.N):
        t_vec = np.append(t_vec, [[data_1.cls]], axis=0)
    for _ in range(data_2.N):
        t_vec = np.append(t_vec, [[data_2.cls]], axis=0)

    for _ in range(10):
        # calc y_vec, R (4.98)
        R = np.zeros((N, N))
        y_vec = np.empty((0, 1))
        for i, phi in enumerate(design_phi):
            R[i, i] = sigmoid(np.dot(w.T, phi)) * (1 - sigmoid(np.dot(w.T, phi)))
            y_vec = np.append(y_vec, [sigmoid(np.dot(w.T, phi))], axis=0)

        # update weights (4.99)
        H = np.dot(np.dot(design_phi.T, R), design_phi)
        w = w - np.dot(np.linalg.inv(H), np.dot(design_phi.T, (y_vec-t_vec)))
        print(w)

    # plot boundary
    plotW(w)

    # plot data
    data_1.scatter('red', '^')
    data_2.scatter('blue', 's')
    # data_out.scatter('blue', 's')
    plt.show()

if __name__ == '__main__':
    __main()
