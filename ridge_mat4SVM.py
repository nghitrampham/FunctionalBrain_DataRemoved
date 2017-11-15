
from numpy.random import uniform
import random
import time
import scipy.io as sio


import numpy as np
import numpy.linalg as LA

import sys

from sklearn.linear_model import Ridge

from utils import create_one_hot_label


class Ridge_Model():

	def __init__(self,class_labels):

		###RIDGE HYPERPARAMETER
		self.lmda = 0.1
		# self.check = 0
		# self.label = class_labels



	def train_model(self,X,Y):

		n = len(np.unique(Y))
		Y = create_one_hot_label(Y,n)
		# print('here is one_hot_vector')
		print('here is lambda for the ridge', self.lmda)
		self.clf = Ridge(alpha=self.lmda)
		self.clf.fit(X, Y)
		# print(self.clf.coef_)


	def eval(self,x):

		# print('here is x_reshape')
		# print(np.shape(x.reshape(1,2)))
		# print('here is lambda', self.lmda)
		# print('here is size of x', np.shape(x))
		dim_x = np.shape(x)[0]
		pre = self.clf.predict(x.reshape(1,dim_x))
		# print('here is pre', pre)


		return np.argmax(pre[0])
