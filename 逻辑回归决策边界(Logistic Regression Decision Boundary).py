import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

Datax = np.array([[3,2],[4,5],[6,5],[7,3],[8,5]])
Datay = np.array([0,0,0,1,1])

x_test = np.array([[10,5]])

model = LogisticRegression()
model.fit(Datax, Datay)

#实行分类
sort = np.argsort(np.where(Datay==0)).flatten()
Data_is_0 = Datax[0:len(sort)]
print(Data_is_0)

x_is_0 = Data_is_0[:,0]
y_is_0 = Data_is_0[:,1]
plt.scatter(x_is_0,y_is_0,color='r')

Data_is_1 = Datax[len(sort):]
x_is_1 = Data_is_1[:,0]
y_is_1 = Data_is_1[:,1]

plt.scatter(x_is_1,y_is_1,color='b')
print(model.coef_)
print(model.intercept_)
#画决策边界

x1_plot = np.linspace(0, 10,100)
x2_plot = (-model.coef_[0][0] * x1_plot - model.intercept_[0]) / model.coef_[0][1]
plt.plot(x1_plot,x2_plot)
#画等高线
axis = [0,13,0,8]
x0,x1 = np.meshgrid(np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1))
X_new = np.c_[x0.ravel(),x1.ravel()]
y_predict = model.predict(X_new)
zz = y_predict.reshape(x0.shape)
print(zz)
custom_cmap = ListedColormap(['#EF9A9B', '#FFF583', '#90CAF9'])

plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
plt.scatter(x_is_0,y_is_0,color='r')
plt.scatter(x_is_1,y_is_1,color='b')

plt.show()

