# -*- coding: utf-8 -*-
"""
Tutorial on how to tune parameters using cross-validation.

In this example, we want to tune the parameters of KNN and find the best parameters 'n_neighbor'.
"""

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# K Nearest Neighbor Classifier, find 10 nearest neighbors
knn = KNeighborsClassifier(n_neighbors=10)

# Performing cross-validation by separaing into 5 different train-validation set.
scores = cross_val_score(knn,X,y,cv=5,scoring='accuracy')
print(scores)
print('Score for n_neighbors=10 : ' + str(scores.mean()))

# Try different value of n_neighbor from 1~31
k_range = range(1,31)
k_scores = []
for k_number in k_range:
    knn = KNeighborsClassifier(n_neighbors=k_number)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
    
plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()