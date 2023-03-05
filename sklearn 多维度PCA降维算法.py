from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

data = np.array([[6, 4,6], [8, 1,5], [10, 3,4], [14, 2,3], [18, 5,8]])
#ax3 = plt.axes(projection='3d')
pca = PCA(n_components=2)
pca.fit(data)
transform_data = pca.transform(data)
print(transform_data)


x = transform_data[:,0]
y = transform_data[:,1]
plt.scatter(x,y)

#dx3 = data[:,0]
#dy3 = data[:,1]
#dz3 = data[:,2]

#ax3.scatter(dx3,dy3,dz3)
plt.show()