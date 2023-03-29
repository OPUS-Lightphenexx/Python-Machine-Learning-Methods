import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

np.random.seed(3949)
wine = load_wine()
x_main_data = load_wine().data
y_label = load_wine().target

from sklearn.model_selection import train_test_split

print(x_main_data)
print(y_label)

#数据分类
x_train, x_test, y_train_label, y_test_label = train_test_split(wine.data,wine.target,test_size=0.3)

#随机森林预测
Random_Forest = RandomForestClassifier()
Random_Forest.fit(x_train,y_train_label)

Random_Forest_score = Random_Forest.score(x_test,y_test_label)
print(Random_Forest_score)

#决策树预测
Decision_tree = DecisionTreeClassifier()
Decision_tree.fit(x_train,y_train_label)

Decision_tree_score = Decision_tree.score(x_test,y_test_label)
print(Decision_tree_score)

score = np.array([Random_Forest_score,Decision_tree_score])
plt.bar(x=range(0,2),height=score)
labels = ["{}".format(i) for i in ['Random Forest','Decision Tree']]
plt.xticks(range(0,2,1),labels)
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Score Comparison Between Random Forest and Decision Tree')
plt.grid()
plt.show()



