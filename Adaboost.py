import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([[1,5],[2,2],[3,1],[4,6],[6,8],[6,5],[7,9],[8,7],[9,8],[10,2]])
y_data_label = np.array([[1],[1],[-1],[-1],[1],[-1],[1],[1],[-1],[-1]])

plt.scatter(x_data[:,0],x_data[:,1],c=y_data_label)
plt.show()

W = np.repeat(1/len(x_data),len(x_data)).reshape(len(x_data),1)#权重
print(W)




