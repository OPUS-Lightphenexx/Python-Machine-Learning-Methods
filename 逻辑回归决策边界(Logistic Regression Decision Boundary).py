import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

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


Data_is_1 = Datax[len(sort):]
x_is_1 = Data_is_1[:,0]
y_is_1 = Data_is_1[:,1]


print(model.coef_)
print(model.intercept_)
#画决策边界

x1_plot = np.linspace(0, 10,100)
x2_plot = (-model.coef_[0][0] * x1_plot - model.intercept_[0]) / model.coef_[0][1]
#plt.plot(x1_plot,x2_plot)
#画等高线
axis = [0,13,0,8]
x0,x1 = np.meshgrid(np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1))
X_reset = np.c_[x0.ravel(),x1.ravel()]
y_predict = model.predict(X_reset)
zz = y_predict.reshape(x0.shape)
print(zz)
custom_cmap = ListedColormap(['#EF9A9B', '#FFF583', '#90CAF9'])

#plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)#没有进行多项式操作
#plt.scatter(x_is_0,y_is_0,color='r')
#plt.scatter(x_is_1,y_is_1,color='b')

#多项式特征
#引入多项式的操作
def PolynomialLogisticRegression(degree):
    return Pipeline([('poly',PolynomialFeatures(degree=degree)),('std_scaler',StandardScaler()),('log_reg',LogisticRegression())])

#等高线函数(这个是方便后面直接调用,直接从网上拿的,前面的由我自己编写)
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0])) * 100).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2])) * 100).reshape(-1, 1)
    )
    x_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(x_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_camp = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, camp=custom_camp)

regre = PolynomialLogisticRegression(2)
regre.fit(Datax,Datay)
plot_decision_boundary(regre,axis)#进行了多项式操作
plt.scatter(x_is_0,y_is_0,color='r')
plt.scatter(x_is_1,y_is_1,color='b')
plt.show()
