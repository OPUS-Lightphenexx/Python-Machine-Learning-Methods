#待处理

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier

x_data = np.array([[1,5],[2,2],[3,1],[4,6],[6,8],[6,5],[7,9],[8,7],[9,8],[10,2]])
y_data_label = np.array([[1],[1],[-1],[-1],[1],[-1],[1],[1],[-1],[-1]]).flatten()

plt.scatter(x_data[:,0],x_data[:,1],c=y_data_label)
plt.show()

W = np.repeat(1/len(x_data),len(x_data)).reshape(len(x_data),1)#权重
print(W)

e = []
for i in range(1,len(x_data)):
    v = i+0.5
    x_data_left = x_data[:i,0]
    x_data_right = x_data[i:,0]
    y_data_left = x_data[:i, 1]
    y_data_right = x_data[i:, 1]

    print(x_data_left,x_data_right)


#算了，先调用sklearn试一试水，找点信心
Model = AdaBoostClassifier()
Train = Model.fit(x_data,y_data_label)
predict = Model.predict(np.array([[1,4]]))
print(predict)
print(Model.estimators_)