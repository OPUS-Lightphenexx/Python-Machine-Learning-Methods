from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RandomForest_Score = []
DecisionTree_Score = []
NaiveBayes_GaussianScore = []
NaiveBayes_MultinomialScore = []
k = 10
for i in range(k):
    np.random.seed(i)
    wine = load_wine()
    x_main_data = load_wine().data
    y_label = load_wine().target

    from sklearn.model_selection import train_test_split

    #print(x_main_data)
    #print(y_label)

    # 数据分类
    x_train, x_test, y_train_label, y_test_label = train_test_split(wine.data, wine.target, test_size=0.3)

    np.random.seed(2904)
    # 高斯分布朴素贝叶斯方法
    model = GaussianNB()
    naive_bayes_training = model.fit(x_train, y_train_label)
    naive_bayes_test_predict = model.predict(x_test)
    Gaussian_test_score = model.score(x_train, y_train_label)
    #print(Gaussian_test_score)

    # 二项分布朴素贝叶斯方法
    model1 = MultinomialNB()
    naive_bayes_training = model1.fit(x_train, y_train_label)
    naive_bayes_test_predict = model1.predict(x_test)
    Multinomial_test_score = model1.score(x_train, y_train_label)
    #print(model.class_prior_)
    #print(Multinomial_test_score)

    # 随机森林预测
    Random_Forest = RandomForestClassifier()
    Random_Forest.fit(x_train, y_train_label)

    Random_Forest_score = Random_Forest.score(x_test, y_test_label)
    #print(Random_Forest_score)

    # 决策树预测
    Decision_tree = DecisionTreeClassifier()
    Decision_tree.fit(x_train, y_train_label)

    Decision_tree_score = Decision_tree.score(x_test, y_test_label)
    #print(Decision_tree_score)

    score = np.array([Random_Forest_score, Decision_tree_score
                         , Multinomial_test_score
                         , Gaussian_test_score])
    labels = ["{}".format(i) for i in ['Random\nForest'
        , 'Decision\nTree'
        , 'Naive Bayes\nGaussian'
        , 'Naive Bayes\nMultinomial']]
    #plt.bar(x=range(0, 4), height=score)
    #plt.xticks(range(0, 4, 1), labels)
    #plt.xlabel('Models')
    #plt.ylabel('Scores')
    #plt.title('Score Comparison Between Random Forest,Decision Tree\n'
              #'Naive Bayes Gaussian and Naive Bayes Multinomial')
    #plt.grid()
    #plt.show()
    RandomForest_Score.append(Random_Forest_score)
    DecisionTree_Score.append(Decision_tree_score)
    NaiveBayes_GaussianScore.append(Gaussian_test_score)
    NaiveBayes_MultinomialScore.append(Multinomial_test_score)

print(RandomForest_Score,
DecisionTree_Score,
Gaussian_test_score,
Multinomial_test_score)

#随机森林评估
plt.plot(RandomForest_Score,label='RandomForestScore')
plt.plot(DecisionTree_Score,label='DecisionTree',color='y')
plt.plot(NaiveBayes_GaussianScore,label='NaiveBayes_Gaussian',color='r')
plt.plot(NaiveBayes_MultinomialScore,label='NaiveBayes_Multinomial',color='green')
plt.xlabel('Seeds')
plt.ylabel('Scores')
plt.title('Score Comparison Between Random Forest,Decision Tree\n'
              'Naive Bayes Gaussian and Naive Bayes Multinomial')
plt.legend(ncol=2,prop = {'size':7})
xticks = range(0,k,1)
plt.xticks(xticks)
plt.grid()
plt.show()






























