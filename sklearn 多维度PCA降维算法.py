from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

data = np.array([[6, 4,6], [8, 1,5], [10, 3,4], [14, 2,3], [18, 5,8]])
ax3 = plt.axes(projection='3d')
pca = PCA(n_components=2)
pca.fit(data)
transform_data = pca.transform(data)
print(transform_data)

#转换后的图像(以二维的形式呈现出来)
#x = transform_data[:,0]
#y = transform_data[:,1]
#plt.scatter(x,y)

#三维演示图像(降维前的数据)
dx3 = data[:,0]
dy3 = data[:,1]
dz3 = data[:,2]

ax3.scatter(dx3,dy3,dz3)
plt.show()

#四维转三维
data_4d = np.array([[6, 4,6,7], [8, 1,5,8], [10, 3,4,6], [14, 2,3,12], [18, 5,8,14]])
pca = PCA(n_components=3)
pca.fit(data)
transform_data_4d = pca.transform(data)
print(transform_data_4d)

x_4d = transform_data_4d[:,0]
y_4d = transform_data_4d[:,1]
z_4d = transform_data_4d[:,2]

#ax3.scatter(x_4d,y_4d,z_4d)
#plt.show()
