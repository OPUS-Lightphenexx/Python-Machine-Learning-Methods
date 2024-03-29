import numpy as np
from collections import Counter


def initial():
    Dataset = np.array([[90,100],[88,90],[85,95],[10,20],[30,40],[50,30]])
    labels = ['A','A','A','D','D','D']
    return Dataset,labels


def KNNClassify(newinput,dataset,labels,k):
    samples = dataset.shape[0]

    minus = np.tile(newinput,(samples,1))-dataset#一一对应相减
    minus_square = minus**2
    squared_distance = np.sum(minus_square,axis=1)
    total_distance = squared_distance**0.5

    sort_index = np.argsort(total_distance)


    for i in range(k):
        count_label = labels[sort_index[i]]
        count = Counter(count_label)
    max_count = 0
    for keys,values in count.items():
        if values>max_count:
            max_count = values
            return_index = keys
        return return_index

dataset,labels = initial()

test_label = np.array([40,50])
k=4

output = KNNClassify(test_label,dataset,labels,3)
print(output)