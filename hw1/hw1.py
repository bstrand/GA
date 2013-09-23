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

def main():
	minK = 1
	maxK = 10
	xvalN = 10

	iris = datasets.load_iris()
	#print "Labels:", repr(np.unique(iris.target_names))

	irisData = iris.data
	irisClasses = iris.target
	irisClassNames = iris.target_names

	# input: full data set
	# output: list of equally sized data splits, length N. Each split has train and test list members, spans the full data set, each slice appears once as test
	# 
	# random permutation of full data set indices
	# define N slices
	# for each slice, create split consisting of test=slice, train=all others
	# 			test=data[a:b]
	# 			train=data[0:a] 
	np.random.seed()
	indices = np.random.permutation(len(irisData)) #len(irisData) = 150
	# print "Train slice: %s" % repr(indices[:-10])
	# print "Test slice: %s" % repr(indices[-10:])

	irisData_train = irisData[indices[-50:]]
	irisClasses_train = irisClasses[indices[-50:]]

	irisData_test = irisData[indices[:-50]]
	irisClasses_test = irisClasses[indices[:-50]]

	print "Training set size: %d, Test set size: %d" % (len(irisData_train), len(irisData_test))


	kVal=0
	for kVal in range (1,10):
		runTrainAndTest(irisData_train,irisClasses_train,irisData_test,irisClasses_test,kVal)

def runTrainAndTest(irisData_train,irisClasses_train,irisData_test,irisClasses_test,kVal=5):
	knn = KNeighborsClassifier(kVal)
	knn.fit(irisData_train, irisClasses_train)
	predictedClasses_test = knn.predict(irisData_test)

	print "Train & test run, K = %d" % kVal
	#print "%d data points were classified incorrectly" % (countErrors(predictedClasses_test, irisClasses_test))

	# print "Accuracy: %d%%, Test set size: %d, Training set size %d" % (
	# 	accuracy(predictedClasses_test, irisClasses_test),
	# 	len(irisData_test),
	# 	len(irisData_train)
	# )

	knnScore = knn.score(irisData_test, irisClasses_test)
	print "KNN score: %.2f" % knnScore
	print "---"

def countErrors(prediction, key):
    diff_count = 0
    for i, j in zip(prediction, key):
        if i != j:
            diff_count +=1
    return diff_count

def accuracy(prediction, key):
    diff_count = 0
    for i, j in zip(prediction, key):
        if i != j:
            diff_count +=1
    pctAccuracy = 100 * (1 - (diff_count / len(key)))
    return pctAccuracy


main()
