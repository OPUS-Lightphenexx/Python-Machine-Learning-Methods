import matplotlib.pyplot as plt
import numpy as np

data = np.array([[6, 4], [8, 1], [10, 3], [14, 2], [18, 5]])
#去中心化

data_norm = data-data.mean(axis = 0)
print('data_norm:',data_norm)
#把x和y分别提取出来
X = data_norm[:,0]
Y = data_norm[:,1]
#计算协方差矩阵
cov = np.cov(data_norm,rowvar=False)

#计算特征值和特征向量
zhi , xiang_liang = np.linalg.eig(cov)
print(zhi,'向量:',xiang_liang)
ev1 = np.array([xiang_liang[:,0]*-1,xiang_liang[:,0]])*100
ev2 = np.array([xiang_liang[:,1]*-1,xiang_liang[:,1]])*100
ev1 = ev1+data.mean(axis=0)
ev2 = ev2+data.mean(axis=0)
print('ev1:',ev1)
print('ev2:',ev2)
line_dot = xiang_liang[:,:1]
Y = np.matmul(data_norm,line_dot)
np.matmul(Y,line_dot.T)#线性代数转置计算方法求投影坐标(超纲了:/)
data_ = np.matmul(Y,line_dot.T)+data.mean(0)


#画图开始
plt.scatter(data_[:,0],data_[:,1],label='restructured data')
plt.plot(ev1[:,0],ev1[:,1],color='r')
plt.plot(ev2[:,0],ev2[:,1])
plt.scatter(data[:,0],data[:,1])

size = 25
plt.plot([-size,size],[0,0],c='black')
plt.plot([0,0],[-size,size],c='black')
plt.xlim(-size,size)
plt.ylim(-size,size)
plt.show()