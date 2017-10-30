# coding:utf-8
"""
パーセプトロンによる2値分類
"""
import sys
from copy import deepcopy
import numpy as np
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

    def scatter(self, color, marker='o'):
        plt.scatter(x=self.phi_vec[:, 1], y=self.phi_vec[:, 2], color=color, marker=marker, alpha=0.4)
        # plt.scatter(x=self.x1, y=self.x2, color=color, marker=marker, alpha=0.4)


class Perceptron:
    def __init__(self):
        self.weight = np.ones((2+1, 1))
        self.eta = 1

    def update(self, phi, t):
        """Update weights
        """
        self.weight += np.array([phi]).T * t


def convertFeatureVector(x):
    phi = np.array([[1, x[0], x[1]]])
    return phi


def plotW(w):
    plt.close()
    plt.figure()
    # separation plane: w1*x1 + w2*x2 + w0 = 0
    x = np.arange(-30, 30, 0.1)
    y = -(w[0] + w[1]*x) / w[2]

    plt.plot(x, y, color='green')


def plotData(data_1, data_2):
    # plot data
    data_1.scatter('red', '^')
    data_2.scatter('blue', 's')
    plt.xlim((-10, 10))
    plt.ylim((-20, 30))
    plt.title('Perceptron Algorithm')


if __name__ == '__main__':
    # データ生成
    # クラス数
    maxcls = 2
    # 総データ数
    N = 20
    # 各クラスのデータの平均・分散
    mean_1 = np.array([0, -4])
    mean_2 = np.array([0, 8])
    cov = np.array([[8, 3], [3, 8]])
    # Dataクラスを用いてデータをN個生成
    data_1 = Data(mean=mean_1, cov=cov, N=int(N/2), cls=-1, maxcls=maxcls)
    data_2 = Data(mean=mean_2, cov=cov, N=int(N/2), cls=1, maxcls=maxcls)

    # non-linear convert
    data_1.phi_vec = np.empty((0, 3))
    data_2.phi_vec = np.empty((0, 3))
    for i, data in enumerate(data_1.x_vec):
        data_1.phi_vec = np.append(data_1.phi_vec, convertFeatureVector(data), axis=0)
    for i, data in enumerate(data_2.x_vec):
        data_2.phi_vec = np.append(data_2.phi_vec, convertFeatureVector(data), axis=0)

    perceptron = Perceptron()

    draw_fg = True

    loop_cnt = 0
    while True:
        loop_cnt += 1
        update_cnt = 0
        for phi in data_1.phi_vec:
            if np.dot(perceptron.weight.T, phi) < 0:
                continue
            update_cnt += 1

            if draw_fg:
                # plot boundary
                plotW(perceptron.weight)
                plotData(data_1, data_2)
                plt.pause(.01)

            perceptron.update(phi, -1)

            if draw_fg:
                # plot boundary
                plotW(perceptron.weight)
                plotData(data_1, data_2)
                plt.pause(.01)

        for phi in data_2.phi_vec:
            if np.dot(perceptron.weight.T, phi) >= 0:
                continue
            update_cnt += 1

            if draw_fg:
                # plot boundary
                plotW(perceptron.weight)
                plotData(data_1, data_2)
                plt.pause(.01)

            perceptron.update(phi, 1)

            if draw_fg:
                # plot boundary
                plotW(perceptron.weight)
                plotData(data_1, data_2)
                plt.pause(.01)

        if update_cnt == 0:
            print("Converged by {} iterations.".format(loop_cnt))
            break
        if loop_cnt >= 1000:
            print("It didn't converge by {} iterations".format(loop_cnt))
            break

    # plot boundary
    plotW(perceptron.weight)
    # plot data
    data_1.scatter('red', '^')
    data_2.scatter('blue', 's')
    plt.xlim((-10, 10))
    plt.ylim((-20, 30))
    plt.title('Perceptron Algorithm')
    plt.show()

