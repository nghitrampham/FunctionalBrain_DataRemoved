import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import inv
from numpy.linalg import det
from sklearn.svm import LinearSVC
from projection_brain import Project2D, Projections


class LDA_Model():

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.001
		self.NUM_CLASSES = len(class_labels)



	def train_model(self,X,Y):

		holder_1 = []
		holder_2 = []
		# holder_3 = []
		for i in range(len(X)):
			if Y[i] == 0:
				holder_1.append(X[i])
			elif Y[i] ==1:
				holder_2.append(X[i])

		state_dim = np.shape(X)[1]
		self.mean_1 = (1/len(holder_1))*sum(holder_1)
		self.mean_1 = self.mean_1.reshape(1,state_dim)
		#make sure to reshape it, numpy array is weird
		self.mean_2 = (1/len(holder_2))*sum(holder_2).reshape(1,state_dim)


		cov = np.zeros([state_dim,state_dim])
		for k in range(len(holder_1)):
			tmp = ((X[k] - self.mean_1[0]).T).dot(X[k]-self.mean_1[0])
			cov = np.add(cov,tmp)
		self.covariance = cov/len(holder_1)
		self.covariance = self.covariance.reshape(state_dim,state_dim)
		#
		# for i in range(len(X)):
		# 	for j in range(self.NUM_CLASSES):




	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		inv_cov = np.linalg.inv(self.covariance + self.reg_cov*np.eye(self.covariance.shape[0]))
		class_1  = -(x-self.mean_1).dot(inv_cov).dot((x-self.mean_1).T)
		class_2 = -(x-self.mean_2).dot(inv_cov).dot((x-self.mean_2).T)
		# class_3 = -(x-self.mean_3).dot(inv_cov).dot((x-self.mean_3).T)
		predict = np.array([[class_1, class_2]])
		return np.argmax(predict[0])
