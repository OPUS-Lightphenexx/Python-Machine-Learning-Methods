from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

train_x = np.array([[90,100,110],[88,90,90],[85,95,95],[10,20,20],[30,40,40],[50,30,30]])
labels = ['A','A','A','D','D','D']

test = np.array([[100,120,300]])

KNN_classify = KNeighborsClassifier(n_neighbors=4)

KNN_classify.fit(train_x,labels)

prediction = KNN_classify.predict(test)

plot_x = train_x[:,0]
plot_y = train_x[:,1]
plot_z = train_x[:,2]
print(plot_z)

plot_x_test = test[0,0]
plot_y_test = test[0,1]
plot_z_test = test[0,2]


ax3 = plt.axes(projection='3d')
ax3.scatter(plot_x,plot_y,plot_z,color='r')
ax3.scatter(plot_x_test,plot_y_test,plot_z_test,color='b',marker='^')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print(prediction)
