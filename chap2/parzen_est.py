# coding:utf-8
# カーネル密度推定法(Parzen推定法)
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


def parzenEstimate(x, x_data, hinge, N):
    # カーネル密度推定法(Parzen推定法)
    # hinge: 平滑化パラメータ
    # カーネル関数: ガウス関数(平均0,分散1)
    for n, x_n in enumerate(x_data):
        if n == 0:
            p = gaussian_pdf((x-x_n)/hinge, 0, 1) * hinge**(-1)
        else:
            p += gaussian_pdf((x-x_n)/hinge, 0, 1) * hinge**(-1)

    return p / N


def generate(N, alpha):
    # データ生成数
    N = 50
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
    x = np.linspace(0, 1, 50)

    # 推定した各区間における密度（ヒストグラム）
    delta = 0.1

    # カーネル密度推定法(Parzen推定法)
    # 確率密度p(x)を推定する
    hinge = 0.07
    p = parzenEstimate(x=x, x_data=x_data, hinge=hinge, N=N)

    # plot
    # 混合ガウス分布の確率密度関数(正解の密度曲線)
    y = alpha*gaussian_pdf(x=x, mu=0.3, var=0.02) + (1-alpha)*gaussian_pdf(x=x, mu=0.75, var=0.01)
    plt.plot(x, y, color='red')

    # 標本点のプロット
    plt.scatter(x=x_data, y=np.zeros(50), color='green', alpha=0.8)

    # 推定した確率密度を棒グラフで図示
    plt.plot(x, p, color='blue')
    plt.title('Kernel Density Estimator (Parzen Estimator) $h=$%.3f' % hinge)
    plt.show()

if __name__ == '__main__':
    __main()
