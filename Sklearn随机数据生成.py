import numpy as np
import matplotlib.pyplot as plt

#以sklearn生成随机数据
from sklearn.datasets import make_gaussian_quantiles
X1, Y1 = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, mean=[1,2],cov=2)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.title('Make-gaussian-quantiles')
plt.show()

from sklearn.datasets import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共3个簇，簇中心在[-1,-1], [1,1], [2,2]， 簇方差分别为[0.4, 0.5, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [1,1], [2,4]], cluster_std=[0.4, 0.5, 0.2])
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.title('make-blobs')
plt.show()

from sklearn.datasets import make_classification
# X1为样本特征，Y1为样本类别输出， 共400个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
np.random.seed(3294892)
X2, Y2 = make_classification(n_samples=400, n_features=2, n_redundant=0,
                             n_clusters_per_class=1, n_classes=3)
plt.scatter(X2[:, 0], X2[:, 1], marker='o',c=Y2)
plt.title('Make-classification')
plt.show()
print(X2)
print(Y2)

#以numpy生成随机数据
np.random.seed(39849)
random_test_1 = np.random.uniform(-3,3,(50,2))
random_test_1_label = np.random.randint(0,2,50)
plt.scatter(random_test_1[:,0],random_test_1[:,1],c=random_test_1_label)
plt.title('Numpy Random')
plt.show()
print(random_test_1)
print(random_test_1_label)









