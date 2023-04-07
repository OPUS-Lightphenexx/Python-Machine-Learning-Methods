import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from sklearn.datasets import make_regression
from sympy import *
import sympy as sym
import math
import random

x2 = np.linspace(-1, 1, 100)
#函数的梯度下降(以二次函数为例子)

def Gradient_Descent2D(theta_start,k,LR,x,y):
    def f1_what(x):
        return x**2+x**3

    x1 = sym.symbols('x')

    def f1prime(x1):  # 导数
        return sym.diff(f1_what(x1), x1)

    DerivativeOfF1 = sym.lambdify((x1), f1prime(x1), 'numpy')  # 算导数值的函数



    n = 0  # 迭代开始

    data_list = []
    x_data_list = []
    x_data_list.append(theta_start)
    while n < k:
        n = n + 1
        theta_start = theta_start - LR * DerivativeOfF1(theta_start)
        x_data_list.append(theta_start)
    for i in x_data_list:
        data_list.append(f1_what(i))


    data_list.reverse()
    x_data_list.reverse()
    plt.scatter(x_data_list, data_list, color='red')
    plt.title('Gradient Descent Test')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.plot(x, y)
    plt.show()
    return x_data_list,data_list



print(Gradient_Descent2D(1,30,0.1,x2,x2**2+x2**3))
print(Gradient_Descent2D(-0.5,100,0.1,x2,x2**2+x2**3))
