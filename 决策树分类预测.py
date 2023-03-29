import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier()

x_test = np.array([[1,1],[2,2],[0,3],[4,5],[6,8],[4,6]])
y_test = np.array([1,1,0,1,1,0])

tree_clf_class = tree_clf.fit(x_test,y_test)

plt.scatter(x_test[:,0],x_test[:,1],c=y_test)


x_predict = np.array([[2,3]])

y_label_predict = tree_clf.predict(x_predict)
print(y_label_predict)
plt.scatter(x_predict[:,0],x_predict[:,1],c=y_label_predict)
plt.show()

#三维决策树
x_train_3D = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])
y_train_label = np.array([1,1,1,0,0])

#开始训练
tree_classify_3D = tree_clf.fit(x_train_3D,y_train_label)

#开始模型预测
x_test_3D = np.array([[5,6,4]])
tree_classify_test = tree_clf.predict(x_test_3D)
print(tree_classify_test)

#三维画图
x_data = x_train_3D[:,0]
y_data = x_train_3D[:,1]
z_data = x_train_3D[:,2]
ax3 = plt.axes(projection='3d')
ax3.scatter(x_data,y_data,z_data,c=y_train_label)
ax3.scatter(x_test_3D[:,0],x_test_3D[:,1],x_test_3D[:,2],color='red')
plt.show()

#高纬度特征决策树
d_x_test = np.array([[1,1,1,1,1,1],
                     [2,1,2,1,1,1],
                     [2,1,1,1,1,1],
                     [1,1,2,1,1,1],
                     [3,1,1,1,1,1],
                     [1,2,1,1,2,2],
                     [4,2,1,1,2,2],
                     [4,2,1,1,2,1],
                     [4,2,2,1,2,1],
                     [1,3,3,3,3,2],
                     [3,3,3,2,3,1],
                     [3,1,2,2,3,2],
                     [1,2,2,3,1,1],
                     [3,2,2,3,1,1],
                     [2,2,1,1,2,2],
                     [3,1,1,2,3,1],
                     [1,1,2,2,2,1]])
d_y_test_label = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
plt.scatter(d_x_test[:,0],d_x_test[:,1],c=d_y_test_label)
plt.show()

tree_model_test1 = tree_clf.fit(d_x_test,d_y_test_label)

x_test_111 = np.array([[2,4,3,2,4,5]])
new_predict = tree_clf.predict(x_test_111)
print(new_predict)

from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(tree_model_test1,out_file=None)
graph = graphviz.Source(dot_data)
graph.render('决策树画图',view=False)

#机器学习分类
from sklearn.datasets import make_blobs
np.random.seed(48782)
X, y = make_blobs(n_samples=1000, n_features=1, centers=[[-1,-1], [1,1]], cluster_std=[0.4, 0.5])
plt.scatter(X[:, 0], X[:, 1],c=y)
plt.title('make-blobs')
plt.show()

x_data_1_d = X
y_1_d_label = y

tree_model_test2 = tree_clf.fit(x_data_1_d,y_1_d_label)

dot_data_ML = tree.export_graphviz(tree_model_test2,out_file=None)
graph_ML = graphviz.Source(dot_data_ML)
graph_ML.render('机器学习决策树画图sklearn',view=True)
