
import random
import time


import numpy as np
import numpy.linalg as LA


from numpy.linalg import inv
from numpy.linalg import det

from projection_brain import Project2D, Projections


class QDA_Model():

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.01
		self.NUM_CLASSES = len(class_labels)



	def train_model(self,X,Y):

		holder_1 = []
		holder_2 = []
		for i in range(len(X)):
			if Y[i] == 0:
				holder_1.append(X[i])
			elif Y[i] ==1:
				holder_2.append(X[i])

		state_dim = np.shape(X)[1];
		self.mean_1 = (1/len(holder_1))*sum(holder_1)
		# print('here is shape of X', state_dim )
		self.mean_1 = self.mean_1.reshape(1,state_dim )
		#make sure to reshape it, numpy array is weird
		self.mean_2 = (1/len(holder_2))*sum(holder_2).reshape(1,state_dim )

		cov_1 = np.zeros([np.shape(X)[1],np.shape(X)[1]])
		cov_2 = np.zeros([state_dim,state_dim])
		for k in range(len(X)):
			if Y[k]==0:
				tmp = ((X[k] - self.mean_1[0]).T).dot(X[k]-self.mean_1[0])
				cov_1 = np.add(cov_1,tmp)
			elif Y[k] ==1:
				tmp = ((X[k] - self.mean_2[0]).T).dot(X[k]-self.mean_2[0])
				cov_2 = np.add(cov_2,tmp)

		# print('here is size of cov_1 and cov_2', np.shape(cov_1), np.shape(cov_2))

		self.cov_class1= (cov_1/len(holder_1)).reshape(state_dim ,state_dim )
		self.cov_class2 = (cov_2/len(holder_2)).reshape(state_dim ,state_dim )



	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		inv_cov_1 = np.linalg.inv(self.cov_class1 + self.reg_cov*np.eye(self.cov_class1.shape[0]))
		inv_cov_2 = np.linalg.inv(self.cov_class2 + self.reg_cov*np.eye(self.cov_class2.shape[0]))
		class_1  = -(x-self.mean_1).dot(inv_cov_1).dot((x-self.mean_1).T) - np.log(np.abs(self.cov_class1.shape[0]))
		class_2 = -(x-self.mean_2).dot(inv_cov_2).dot((x-self.mean_2).T) - np.log(np.abs(self.cov_class2.shape[0]))
		predict = np.array([[class_1, class_2]])
		return np.argmax(predict[0])
