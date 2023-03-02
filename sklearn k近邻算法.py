from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import numpy as np

#分类训练
train_x_test = np.array([[90,100],[88,90],[85,95],[10,20],[30,40],[50,30],[80,56],[67,80],[90,90]])
labels_test = ['A','A','A','D','D','D','A','D','A']
sort = np.argsort(labels_test)
sort2 = sort[0:labels_test.count('A')]
sort3 = sort[labels_test.count('A'):]
train_x_test_list = np.zeros(labels_test.count('A'))
train_A_X_list = []
train_A_Y_list = []
for i in sort2:
     train_A_X = train_x_test[i,0]
     train_A_Y = train_x_test[i,1]
     train_A_X_list.append(train_A_X)
     train_A_Y_list.append(train_A_Y)


train_D_X_list = []
train_D_Y_list = []
for b in sort3:
     train_D_X = train_x_test[b,0]
     train_D_Y = train_x_test[b,1]
     train_D_X_list.append(train_D_X)
     train_D_Y_list.append(train_D_Y)


#plt.scatter(train_D_X_list,train_D_Y_list,color='r')
#plt.scatter(train_A_X_list,train_A_Y_list,color='b')
#plt.show()





#sklearn 进行训练
train_x = np.array([[90,100],[88,90],[85,95],[10,20],[30,40],[50,30]])
labels = ['A','A','A','D','D','D']

test = np.array([[40,50]])

KNN_classify = KNeighborsClassifier(n_neighbors=4)

KNN_classify.fit(train_x,labels)

prediction = KNN_classify.predict(test)

print(prediction)

sort1 = np.argsort(labels)
sort3 = sort1[0:labels.count('A')]
sort4 = sort1[labels.count('A'):]
train_A_X_list1 = []
train_A_Y_list1 = []
for i in sort3:
     train_A_X1 = train_x[i,0]
     train_A_Y1 = train_x[i,1]
     train_A_X_list1.append(train_A_X1)
     train_A_Y_list1.append(train_A_Y1)


train_D_X_list1 = []
train_D_Y_list1 = []
for b in sort4:
     train_D_X1 = train_x[b,0]
     train_D_Y1 = train_x[b,1]
     train_D_X_list1.append(train_D_X1)
     train_D_Y_list1.append(train_D_Y1)

plt.scatter(train_A_X_list1,train_A_Y_list1,color='r',)
plt.scatter(train_D_X_list1,train_D_Y_list1,color='blue')
plt.scatter(test[:,0],test[:,1],color='green')
plt.grid('on')
plt.show()


