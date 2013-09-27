# GA-DS Section 2, Homework #1
# Student: Brant Strand
#
# Create a KNN classifier using scikit-learn (file name GA_homework/hw1/hw1.py).
# Use the iris dataset.
# Run cross validation for various values of k.
# Push to github
# Bonus:
# Organize your script into functions.
# Extra Bonus:
# Derive your own hand-written KNN classifier and compare results to scikit-learn's KNN classifier results.

from __future__ import division
import numpy as np
import itertools
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

def main():
	min_K = 1
	max_K = 10
	k_val=0
	cv_passes = 5

	iris = datasets.load_iris()

#	np.random.seed()
#	indices = np.random.permutation(len(iris.data)) #len(irisData) = 150

	n_folds = 10
	skfold = StratifiedKFold(iris.target, n_folds)
	kfold = KFold(len(iris.target), n_folds)

	for k_val in range (min_K,max_K):
		estimate_KNN_accuracy(k_val, iris.data, iris.target, skfold)

def estimate_KNN_accuracy(k_val, data, target, cv=None):
	knn = KNeighborsClassifier(k_val)
	scores = cross_val_score(knn, data, target, cv=cv)
	print("Accuracy for K=%d: %0.2f (+/- %0.2f)" % (k_val, scores.mean(), scores.std() * 2))

main()
