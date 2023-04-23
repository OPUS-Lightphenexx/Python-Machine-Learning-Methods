import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[6, 4], [8, 1], [10, 3], [14, 2], [18, 5]])
y = np.array([[7], [9], [13], [17.5], [18]])
model = LinearRegression()
model.fit(X, y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]
predictions = model.predict(X)
print(model.coef_)

ax3 = plt.axes(projection='3d')
x1 = X[:,0]
y1 = X[:,1]
z1 = y[:,0]
print(x1)

x3 = X[:,0]
y3 = X[:,1]
X,Y = np.meshgrid(x3,y3)
z4 = model.intercept_ + X*model.coef_[0][0]+ Y*model.coef_[0][1]
ax3.scatter(x1,y1,z1,color='r')
ax3.plot_surface(X,Y,z4)
plt.show()

#测试(预测x的值)
X = [[6, 4], [8, 1], [10, 3], [14, 2], [18, 5]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()
model.fit(X, y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]
predictions = model.predict(X)
print(predictions)
