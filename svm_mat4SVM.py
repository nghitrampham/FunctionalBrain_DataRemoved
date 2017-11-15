from numpy.random import uniform
import random
import time

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys

from sklearn.svm import LinearSVC
from projection_brain import Project2D, Projections


class SVM_Model():

	def __init__(self,class_labels,projection=None):

		###SLACK HYPERPARAMETER
		self.C = 0.1


	def train_model(self,X,Y):
		''''
		run linear SVC

		'''
		# print('here is the C', self.C)
		self.clf = LinearSVC(random_state=0, C = self.C)
		self.clf.fit(X,Y)

	def eval(self,x):
		''''
		Evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		# print('here is size of x', np.shape(x))
		# pre = self.clf.predict(x.reshape(1,len(x)))

		pre = self.clf.predict(x.reshape(1,len(x)))
		# pre = self.clf.predict(x.reshape(1,2))
		# print(pre)
		return pre[0]
