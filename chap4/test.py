# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import sys
from scipy.stats import norm

sys.path.append('../sampledata')
import generate_data as gen


mean = np.array([0, 0])
cov = np.array([[2, -1.5], [-1.5, 2]])
gauss2d = gen.GenerateRandomNoize2D(mean=mean, cov=cov, n=50)
gauss2d.generate()

plt.scatter(x=gauss2d.x, y=gauss2d.y, alpha=0.4, color='red')
plt.xlim((-4, 10))
plt.ylim((-4, 10))
plt.show()