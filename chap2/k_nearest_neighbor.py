# coding:utf-8
# K近傍法(K-nearest neighbor method)
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import sys
from scipy.stats import norm

sys.path.append('../sampledata')
import generate_data as gen


def pr(p):

    '''
    return True with probability p
    return False with probability 1-p
    '''

    if rd.random() < p:
        return True
    else:
        return False


def gaussian_pdf(x, mu, var):
    # 1次元ガウス分布 確率密度関数(PDF)
    # mu: 平均, var: 分散
    tmp1 = 1.0/(np.sqrt(2*np.pi*var))
    tmp2 = np.exp(-(x-mu)**2 / (2*var))
    return tmp1*tmp2


def KNearestNeighbor(x, x_data, K, N):
    # K近傍法(k-nearest neighbor)
    # K: 小球の体積Vに含まれるデータ点の最大数
    # x_data: データ点
    # x_obj: 推定したい点 x
    p = np.zeros(N)
    for n, x_obj in enumerate(x):
        # r: 小球の半径
        r = 0
        while True:
            # 小球(1次元)の半径を広げつつ，範囲内に入る点の数を確認
            r += 0.0001
            k = len(set(x_data[x_obj-r <= x_data]) & set(x_data[x_data <= x_obj+r]))
            if k >= K:
                break

        V = 2*r
        p[n] += K/(N*V)

    return p


def generate(N, alpha):
    # データ生成数
    N = N
    gen_noize1 = gen.GenerateRandomNoize(0.3, 0.02, N)
    gen_noize2 = gen.GenerateRandomNoize(0.75, 0.01, N)
    gen_noize1.generate()
    gen_noize2.generate()
    x1 = gen_noize1.value
    x2 = gen_noize2.value

    alpha = 0.4
    x_data = np.array([])

    # 確率的にデータ生成
    for i in range(N):
        if pr(alpha):
            x_data = np.append(x_data, x1[i])
        else:
            x_data = np.append(x_data, x2[i])

    return x_data


def __main():
    # データ生成数, 混合分布の重み
    N = 50
    alpha = 0.4
    x_data = generate(N, alpha)
    x = np.linspace(0, 1, N)

    # K近傍法
    # 確率密度p(x)を推定する
    K = 5
    p = KNearestNeighbor(x=x, x_data=x_data, K=K, N=N)

    # plot
    # 混合ガウス分布の確率密度関数(正解の密度曲線)
    y = alpha*gaussian_pdf(x=x, mu=0.3, var=0.02) + (1-alpha)*gaussian_pdf(x=x, mu=0.75, var=0.01)
    plt.plot(x, y, color='red')
    # plt.ylim((0, 5))

    # 標本点のプロット
    plt.scatter(x=x_data, y=np.zeros(N), color='green', alpha=0.8)

    # 推定した確率密度を棒グラフで図示
    plt.plot(x, p, color='blue')
    plt.title('Kernel Density Estimator (Parzen Estimator) $K=$%d' % K)
    plt.show()

if __name__ == '__main__':
    __main()
