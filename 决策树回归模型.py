from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共3个簇，簇中心在[-1,-1], [1,1], [2,2]， 簇方差分别为[0.4, 0.5, 0.2]
X, y = make_blobs(n_samples=1000, n_features=1, centers=[[-1,-1], [1,1]], cluster_std=[0.4, 0.5])


x_data = np.array(X[:,0]).reshape(-1,1)
y_data = np.array(X[:,1]).reshape(-1,1)
X_test = np.arange(-2, 4, 0.01).reshape(-1,1)


model = DecisionTreeRegressor(max_depth=100)
model.fit(x_data,y_data)
predict = model.predict(X_test)
print(model.score(x_data,y_data))

plt.plot(X_test,predict)
plt.scatter(X[:, 0], X[:, 1], marker='o',c=y)
plt.title('Decision Tree Regressor')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

#线性数据
X1 = np.array([[6, 4], [8, 1], [10, 3], [14, 2], [18, 5]])
x_lin_data = np.array(X1[:,0]).reshape(-1,1)
y_lin_data = np.array(X1[:,1]).reshape(-1,1)

model1 = DecisionTreeRegressor(max_depth=6)
model1.fit(x_lin_data,y_lin_data)
X_test_again1 = np.arange(5,18, 0.01).reshape(-1,1)
predict1 = model1.predict(X_test_again1)

model2 = DecisionTreeRegressor(max_depth=2)
model2.fit(x_lin_data,y_lin_data)
X_test_again2 = np.arange(5,18, 0.01).reshape(-1,1)
predict2 = model2.predict(X_test_again2)

model3 = DecisionTreeRegressor(max_depth=3)
model3.fit(x_lin_data,y_lin_data)
X_test_again3 = np.arange(5,18, 0.01).reshape(-1,1)
predict3 = model3.predict(X_test_again3)

model4 = DecisionTreeRegressor(max_depth=4)
model4.fit(x_lin_data,y_lin_data)
X_test_again4 = np.arange(5,18, 0.01).reshape(-1,1)
predict4 = model4.predict(X_test_again4)

plt.subplot(221)
plt.scatter(x_lin_data,y_lin_data)
plt.plot(X_test_again2,predict2,label='Max_Depth=2')
plt.title('Decision Tree Regressor')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.subplot(222)
plt.scatter(x_lin_data,y_lin_data)
plt.plot(X_test_again1,predict1,label='Max_Depth=6',alpha=1)
plt.title('Decision Tree Regressor')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.subplot(223)
plt.scatter(x_lin_data,y_lin_data)
plt.plot(X_test_again3,predict3,label='Max_Depth=3',alpha=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.subplot(224)
plt.scatter(x_lin_data,y_lin_data)
plt.plot(X_test_again4,predict4,label='Max_Depth=4',alpha=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()

plt.show()