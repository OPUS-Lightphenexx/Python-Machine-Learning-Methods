import numpy as np
from sklearn.linear_model import Perceptron
import pandas as pd
import matplotlib.pyplot as plt

#M-P模型
#一维向量模型
x_i = np.array([1,3,5,8]).reshape(-1,1)
r_i = np.array([0,0,1,1])
w_i = np.array([1,1,1,1])
r = 0


h = 13
def func(x_i,w_i,h):
    y = np.dot(x_i,w_i)
    if y>h:
        return 1
    else:
        return 0


#end一下

#用sklearn节省调参过程
#利用pandas创建表格
Dict = {'x1':[1,4,3,2,5,6,7,7],'x2':[4,8,6,3,5,6,13],'label':[0,0,0,1,1,1,1]}
Data_table = pd.DataFrame(Dict)
print(Data_table)
perceptron = Perceptron(fit_intercept=True, max_iter=1000, shuffle=True)
x_ip = Data_table[['x1','x2']]
print(x_ip)
r_ip = Data_table['label']

perceptron.fit(x_ip,r_ip)
w_ip = perceptron.coef_[0]
print(perceptron.coef_[0])
b = perceptron.intercept_
print(b)

plt.scatter(Data_table['x1'],Data_table['x2'],c=Data_table['label'])
xs = np.linspace(0,7,100)
plt.plot(xs,(w_ip[0]*xs+b)/(-w_ip[1]))
plt.show()




        





