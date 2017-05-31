# coding:utf-8
# フィッシャーの線形識別

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


def calcM(x):
    # データ点の平均ベクトルを計算
    m = np.sum(x, axis=0) / len(x)
    m = np.reshape(m, (-1, 1))
    return m


def calcS_b(m1, m2):
    # クラス間共分散行列S_Bを計算 (4.27)式
    S_b = np.dot(m2-m1, (m2-m1).T)
    return S_b


def calcS_w(data_list):
    # 総クラス内共分散行列S_wを計算 (多クラスの場合はdataをリストに格納して渡せばOK)
    # (4.28)式
    for i, data in enumerate(data_list):
        if i == 0:
            S_w = np.dot(data.x_vec.T-data.m, (data.x_vec.T-data.m).T)
        else:
            S_w += np.dot(data.x_vec.T-data.m, (data.x_vec.T-data.m).T)

    return S_w


def calcW(S_w, m1, m2):
    # 射影する際の重みwを計算する (4.30)式
    # ベクトルwと(m2-m1)は平行, wは射影する軸方向のベクトル
    w = np.dot(np.linalg.inv(S_w), (m2-m1))
    return w


def plotW(w, b):
    # ベクトル w = (w[0], w[1])方向の直線をプロットする
    # 傾き a = w[1] / w[0]
    # 切片 b (パラメータ)
    a = w[1] / w[0]
    x = np.arange(-15, 15, 0.1)
    y = a*x + b
    plt.plot(x, y, color='green')


def __main():
    # クラス数
    maxcls = 2
    # 各クラスのデータの平均・分散
    mean_1 = np.array([0, 0])
    mean_2 = np.array([0, 8])
    cov = np.array([[5, -1.6], [-1.6, 5]])
    # Dataクラスを用いてデータをN個生成
    data_1 = Data(mean=mean_1, cov=cov, N=100, cls=1, maxcls=maxcls)
    data_2 = Data(mean=mean_2, cov=cov, N=100, cls=2, maxcls=maxcls)

    data_1.m = calcM(data_1.x_vec)
    data_2.m = calcM(data_2.x_vec)

    S_b = calcS_b(m1=data_1.m, m2=data_2.m)
    S_w = calcS_w(data_list=[data_1, data_2])
    w = calcW(S_w=S_w, m1=data_1.m, m2=data_2.m) * 1000
    print(w)

    # 1次元に射影 (4.20)式
    y1 = np.dot(w.T, data_1.x_vec.T)
    y1 = np.reshape(y1, -1)
    y2 = np.dot(w.T, data_2.x_vec.T)
    y2 = np.reshape(y2, -1)

    # データ点プロット
    data_1.scatter('red', '^')
    data_2.scatter('blue', 's')
    plt.scatter(x=data_1.m[0], y=data_1.m[1], color='black', marker='x')
    plt.scatter(x=data_2.m[0], y=data_2.m[1], color='black', marker='x')

    # 射影軸方向 w の直線のプロット
    plotW(w, -25)
    plt.xlim((-10, 10))
    plt.ylim((-15, 20))
    plt.title('2-Class Fisher\'s linear discriminant')
    plt.show()

    # 射影した後のヒストグラムのプロット
    plt.hist(x=y1, bins=15, alpha=0.5, color='red')
    plt.hist(x=y2, bins=15, alpha=0.5, color='blue')
    plt.show()

if __name__ == '__main__':
    __main()