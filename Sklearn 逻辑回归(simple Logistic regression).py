import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

Datax = np.array([[3,2],[4,5],[6,5],[7,3],[8,5]])
Datay = np.array([0,0,0,1,1])

x_test = np.array([[10,5]])

model = LogisticRegression()
model.fit(Datax, Datay)


y_test_pred = model.predict(X=x_test)

print(y_test_pred)#打印出标签对应分类

print('变量参数:',model.coef_)   # 查看变量参数
print('常数参数项:',model.intercept_)   # 查看常数参数项
print('测试级概率预测:',model.predict_proba(x_test))
prob = model.predict_proba(x_test)
test_prob_1 = prob[:,1]
print('测试集为标签1的概率:',test_prob_1)

train_prob = model.predict_proba(Datax)
#print(train_prob)
train_prob_1 = train_prob[:,1]
print('训练集为标签1的概率:',train_prob_1)

x1 = np.array([Datax[:,0]])
print(x1)
x2 = np.array([Datax[:,1]])

x3 = np.array([x_test[:,0]])
x4 = np.array([x_test[:,1]])

print('查看变量参数:',model.coef_)   # 查看变量参数
print('查看常数参数项:',model.intercept_)   # 查看常数参数项

func_x = model.intercept_ + x1*model.coef_[:,0] + x2*model.coef_[:,1]
func_test_x = model.intercept_+x3*model.coef_[:,0] + x4*model.coef_[:,1]
print(func_x)
print(func_test_x)
func_test_y = np.array([test_prob_1])

#画出sigmoid函数
x = np.linspace(-5,5,100)
y = 1/(1+np.exp(-x))

#画出竖线
x1 = np.zeros(100)
y1 = np.linspace(0,1,100)


#画图开始
plt.scatter(func_x,train_prob_1,color='r')
plt.scatter(func_test_x,func_test_y,color='y')
plt.plot(x,y)
plt.plot(x1,y1,color='black')

plt.show()

#对损失函数进行分析从而达到参数优化的效果这里不作讨论


