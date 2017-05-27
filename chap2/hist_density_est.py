# coding:utf-8
# ヒストグラム密度推定法(HDE)
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
    # var: 分散
    tmp1 = 1.0/(np.sqrt(2*np.pi*var))
    tmp2 = np.exp(-(x-mu)**2 / (2*var))
    return tmp1*tmp2


def HDE(x, delta):
    # delta:ヒストグラムの幅
    # n:幅deltaの区間内におけるxの数
    n = np.zeros(int(1/delta))

    for i in range(int(1/delta)):
        for j in x:
            if i*delta < j and j < (i+1)*delta:
                n[i] += 1

    # p:各区間の確率密度(2.241)
    p = n / (len(x)*delta)
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

    # 推定した各区間における密度（ヒストグラム）
    delta = 0.1

    # ヒストグラム推定(HDE)
    p = HDE(x_data, delta)

    # plot
    x = np.linspace(0, 1, N)
    # 混合ガウス分布の確率密度関数(正解の密度曲線)
    y = alpha*gaussian_pdf(x=x, mu=0.3, var=0.02) + (1-alpha)*gaussian_pdf(x=x, mu=0.75, var=0.01)
    plt.plot(x, y, color='red')

    # 標本点のプロット
    plt.scatter(x=x_data, y=np.zeros(N), color='green', alpha=0.8)

    # 推定した密度を棒グラフで図示
    # arangeの関係上、終端はdeltaで補正
    plt.bar(left=np.arange(0, delta*int(1/delta), delta),
            height=p,
            alpha=0.4,
            width=delta,
            align='edge',
            edgecolor='black',
            linewidth=2)
    plt.title('Histogram Density Estimation ($\Delta=$%.2f)' % delta)
    plt.show()

if __name__ == '__main__':
    __main()
