from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

Dict = {'x1':[1,4,3,2,5,6,6,7],'x2':[4,8,6,3,5,6,13,14]}

X_Data = np.array([1,4,3,2,5,6]).reshape(-1,1)
Y_Data = np.array([4,8,6,3,5,6]).reshape(-1,1)

model1 = SVR(kernel='linear')
model1.fit(X_Data,Y_Data)
X_test = np.linspace(0,8,100).reshape(-1,1)
y = model1.predict(X_test)
model1_score = model1.score(X_Data,Y_Data)

model2 = LinearRegression()
model2.fit(X_Data,Y_Data)
y1 = model2.predict(X_test)
model2_score = model2.score(X_Data,Y_Data)

plt.subplot(221)
plt.plot(X_test,y)
plt.scatter(X_Data,Y_Data)
plt.grid()
plt.subplot(222)
plt.plot(X_test,y1)
plt.scatter(X_Data,Y_Data)
i = model2_score
plt.text(0,7,'{}'.format(i),fontsize=12)
plt.grid()
plt.show()

