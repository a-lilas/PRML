import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class GenerateRandomNoize:
    '''
    ガウス分布に従って乱数を生成するオブジェクト
    '''

    def __init__(self, mean, sd, n):
        self.mean = mean
        self.sd = sd
        self.n = n

    def generate(self):
        # 正規ガウス分布
        # loc:期待値、scale:分散、size:データ数
        self.value = np.random.normal(loc=self.mean, scale=self.sd**2, size=self.n)


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

    gen_noize = GenerateRandomNoize(mean=0, sd=0.3, n=20)
    gen_noize.generate()
    noize = gen_noize.value

    uniform = GenerateUniform(low=0, high=1.0, n=20)
    uniform.generate()
    x = uniform.value

    # cos曲線からデータをランダムで抽出
    plt.scatter(x=x, y=(np.sin(2*np.pi*x)+noize))
    plt.show()
