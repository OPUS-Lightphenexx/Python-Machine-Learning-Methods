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

model3 = SVR(kernel='rbf')
model3.fit(X_Data,Y_Data)
X_test = np.linspace(0,8,100).reshape(-1,1)
y2 = model3.predict(X_test)
model3_score = model3.score(X_Data,Y_Data)

model2 = LinearRegression()
model2.fit(X_Data,Y_Data)
y1 = model2.predict(X_test)
model2_score = model2.score(X_Data,Y_Data)

model4 = SVR(kernel='sigmoid')
model4.fit(X_Data,Y_Data)
y3 = model4.predict(X_test)
model4_score = model4.score(X_Data,Y_Data)

plt.subplot(221)
plt.plot(X_test,y)
plt.scatter(X_Data,Y_Data)
plt.title('SVR(Linear)')
plt.grid()
plt.subplot(222)
plt.plot(X_test,y1)
plt.scatter(X_Data,Y_Data)
i = model2_score
plt.title('Linear Regression')
plt.grid()
plt.subplot(223)
plt.title('SVR(RBF)',fontsize=8)
plt.scatter(X_Data,Y_Data)
plt.plot(X_test,y2)
plt.grid()
plt.subplot(224)
plt.scatter(X_Data,Y_Data)
plt.plot(X_test,y3)
plt.title('SVR(Sigmoid)',fontsize=8)
plt.grid()
plt.show()

def Draw_plot_SVR_Sigmoid(x:list,y:list):
    from sklearn.svm import SVR
    main_model = SVR(kernel='sigmoid')
    x_data = np.array(x).reshape(-1,1)
    y_data = np.array(y).reshape(-1,1)
    plt.scatter(x_data,y_data)
    main_model.fit(x_data,y_data)
    X_test = np.linspace(0, 8, 100).reshape(-1, 1)
    predict_data = main_model.predict(X_test)
    plt.plot(X_test,predict_data)
    plt.grid()
    plt.title('SVR(Sigmoid)')
    plt.show()

def Draw_plot_SVR_Linear(x:list,y:list):
    from sklearn.svm import SVR
    main_model = SVR(kernel='linear')
    x_data = np.array(x).reshape(-1,1)
    y_data = np.array(y).reshape(-1,1)
    plt.scatter(x_data,y_data)
    main_model.fit(x_data,y_data)
    X_test = np.linspace(0, 8, 100).reshape(-1, 1)
    predict_data = main_model.predict(X_test)
    plt.plot(X_test,predict_data)
    plt.grid()
    plt.title('SVR(Linear)')
    plt.show()

print(Draw_plot_SVR_Sigmoid([1,2,3,4,5,6,7],[4,5,4,6,3,2,6]))
print(Draw_plot_SVR_Linear([1,2,3,4,5,6,7],[4,5,4,6,3,8,6]))





