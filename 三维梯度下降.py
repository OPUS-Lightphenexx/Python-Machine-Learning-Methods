import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from sklearn.datasets import make_regression
from sympy import *
import sympy as sym
import math
import random

x3 = np.linspace(-1, 1, 100)

def Gradient_Descent3D(theta_startx,theta_starty,k,LR):
    def f1_what(x,y):
        return x**2+y**2

    x1 = sym.symbols('x')
    y1 = sym.symbols('y')

    def f1primex(x1):  # x偏导导数
        return sym.diff(f1_what(x1,0), x1)

    def f1primey(y1):
        return sym.diff(f1_what(0,y1), y1)

    print(f1primex(x1))
    print(f1primey(y1))


    DerivativeOfF1 = sym.lambdify((x1), f1primex(x1), 'numpy')  # 算导数值的函数
    DerivativeOfF2 = sym.lambdify((y1), f1primey(y1), 'numpy')



    n = 0  # 迭代开始
    x_y_array = np.array([theta_startx,theta_starty])
    print(np.array([DerivativeOfF1(x_y_array[0]),DerivativeOfF2(x_y_array[1])]))

    data_list = []
    x_data_list = []
    scatter_x_data_list = []
    scatter_y_data_list = []
    while n < k:
        n = n + 1
        x_y_array = x_y_array - LR * np.array([DerivativeOfF1(x_y_array[0]),DerivativeOfF2(x_y_array[1])])
        x_data_list.append(x_y_array)
    for i,j in x_data_list:
        data_list.append(f1_what(i,j))
        scatter_x_data_list.append(i)
        scatter_y_data_list.append(j)

    print(x_data_list)
    print(data_list)
    print(scatter_x_data_list)
    print(scatter_y_data_list)


    data_list.reverse()
    x_data_list.reverse()
    scatter_y_data_list.reverse()
    scatter_x_data_list.reverse()
    ax3 = plt.axes(projection='3d')
    x = np.arange(-5, 5, 0.5)
    y = np.arange(-5, 5, 0.5)
    X, Y = np.meshgrid(x, y)
    z = X ** 2 + Y ** 2
    ax3.scatter(scatter_x_data_list, scatter_y_data_list,data_list, color='red')
    ax3.plot_surface(X, Y, z, label='Test Function')
    ax3.scatter(4,4,32,color='green')
    plt.show()


print(Gradient_Descent3D(4,4,10,0.2))