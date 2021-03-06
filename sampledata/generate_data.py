import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class GenerateRandomNoize:
    '''
    1次元ガウス分布に従ってデータ点を生成するオブジェクト
    sd : 標準偏差
    '''

    def __init__(self, mean, var, n):
        self.mean = mean
        self.sd = np.sqrt(var)
        self.var = var
        self.n = n

    def generate(self):
        # 正規ガウス分布
        # loc:期待値、scale:分散、size:データ数
        self.value = np.random.normal(loc=self.mean, scale=np.sqrt(self.var), size=self.n)


class GenerateRandomNoize2D:
    '''
    2次元ガウス分布に従ってデータ点を生成するオブジェクト
    mean    : x,yの平均(np.array([*,*]))
    cov     : x,yの分散共分散行列
    '''

    def __init__(self, mean, cov, n):
        self.mean = mean
        self.cov = cov
        self.n = n

    def generate(self):
        # 正規ガウス分布
        # loc:期待値、cov:分散共分散行列、size:データ数
        self.x, self.y = np.random.multivariate_normal(mean=self.mean,
                                                       cov=self.cov,
                                                       size=self.n).T


class GenerateUniform:
    '''
    一様分布にしたがって実数を生成するオブジェクト
    '''

    def __init__(self, low, high, n):
        self.low = low
        self.high = high
        self.n = n

    def generate(self):
        # 一様分布
        # low:最小値、high:最大値、size:データ数
        self.value = np.random.uniform(low=self.low, high=self.high, size=self.n)


if __name__ == '__main__':

    gen_noize = GenerateRandomNoize(mean=0, sd=0.3**2, n=20)
    gen_noize.generate()
    noize = gen_noize.value

    uniform = GenerateUniform(low=0, high=1.0, n=20)
    uniform.generate()
    x = uniform.value

    # cos曲線からデータをランダムで抽出
    plt.scatter(x=x, y=(np.sin(2*np.pi*x)+noize))
    plt.show()
