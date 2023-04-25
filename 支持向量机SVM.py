import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import LinearSVC

#沿用感知机模型里面的数据
Dict = {'x1':[1,4,3,2,5,6,6,7],'x2':[4,8,6,3,5,6,13,14],'label':[0,0,0,1,1,1,0,0]}
Data_table = pd.DataFrame(Dict)


plt.scatter(Data_table['x1'],Data_table['x2'],c=Data_table['label'])


svm_model = LinearSVC()
svm_model.fit(Data_table[['x1','x2']],Data_table['label'])
w_i = svm_model.coef_[0]
b = svm_model.intercept_
print(w_i)
print(b)

x = np.linspace(0,7,100)
y = (w_i[0]*x+b)/(-w_i[1])

x1 = np.linspace(0,7,100)
y1 = (w_i[0]*x+b+1)/(-w_i[1])

x2 = np.linspace(0,7,100)
y2 = (w_i[0]*x+b-1)/(-w_i[1])
plt.plot(x1,y1,linestyle='--',color = 'black')
plt.plot(x2,y2,linestyle='--',color = 'black')
plt.plot(x,y)
plt.title('Support Vector Machine Using Sklearn')
plt.xlabel('X')
plt.ylabel('Y')
text1 = r'$w_1=0.6511,w_2=-0.5302$'
text2 = r'$b=0.5674$'
text3 = r'$y_{i}(W^{T}X+b)\geq 1$'
plt.text(0,13,text1,fontsize=17)
plt.text(0,11,text2,fontsize=17)
plt.text(0,9,text3,fontsize = 17)
plt.grid()
plt.show()