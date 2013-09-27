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
	# Test parameters 
	min_K = 1
	max_K = 10
	k_val=0
	n_folds = 10

	# Load Iris data set
	iris = datasets.load_iris()

	print
	print "%d-fold cross validation of KNN estimator for values of K from %d-%d" % (n_folds, min_K, max_K)
	print "-----------------"

	# TODO: KFold CV iterator isn't working, figure out why
	#kfold = KFold(len(iris.target), n_folds=n_folds, indices=True)

	# Generate mapping defining different folds of the data set for CV
	skfold = StratifiedKFold(iris.target, n_folds)

	# For various values of K, run the train & test and evaluate nearest neighbor estimator's accuracy
	for k_val in range (min_K,max_K+1):
		estimate_KNN_accuracy(k_val, iris.data, iris.target, skfold)

	print

def estimate_KNN_accuracy(k_val, data, target, cv=None):
	knn = KNeighborsClassifier(k_val)
	# Train & test the KNN estimator with each fold defined by the provided CV iterator
	scores = cross_val_score(knn, data, target, cv=cv)
	# Print summary results for this value of K
	print("Accuracy for K=%d: %0.2f (+/- %0.2f)" % (k_val, scores.mean(), scores.std() * 2))

main()
