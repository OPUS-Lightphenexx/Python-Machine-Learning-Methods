import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scikitplot as skplt

#原始数据
train_x_test = np.array([[90,100],[88,90],[85,95],[10,20],[30,40],[50,30],[80,56],[67,80],[90,90]])

plot_x = train_x_test[:,0]
plot_y = train_x_test[:,1]

cluster2 = KMeans(n_clusters=2).fit_predict(train_x_test)
cluster3 = KMeans(n_clusters=3).fit_predict(train_x_test)
cluster4 = KMeans(n_clusters=4).fit_predict(train_x_test)
cluster5 = KMeans(n_clusters=6).fit_predict(train_x_test)

#SSE 手寸方法确定k值
SSE_data = []
for i in range(2,10):
    kmeans = KMeans(n_clusters=i)
    result_list = kmeans.fit_predict(train_x_test)
    SSE = kmeans.inertia_
    SSE_data.append(SSE)

#画图
plt.subplot(221)
plt.scatter(plot_x,plot_y,c=cluster2)
plt.grid()
plt.subplot(222)
plt.scatter(plot_x,plot_y,c=cluster3)
plt.grid()
plt.subplot(223)
plt.scatter(plot_x,plot_y,c=cluster5)
plt.grid()
plt.subplot(224)
plt.plot(range(2,10),SSE_data,label='SSE')
plt.grid()
plt.title('SSE')

plt.show()

kmeans = KMeans(random_state=1)

skplt.cluster.plot_elbow_curve(kmeans,train_x_test,cluster_ranges=(1,9))
plt.show()
