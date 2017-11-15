from numpy.random import uniform
import random
import time


import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
import scipy.io as sio
from sklearn.model_selection import train_test_split

import sys


from projection_brain import Project2D, Projections
from confusion_mat import getConfusionMatrixPlot

from ridge_mat4SVM import Ridge_Model
from qda_mat4SVM import QDA_Model
from lda_mat4SVM import LDA_Model
from svm_mat4SVM import SVM_Model


class Model():
	""" Generic wrapper for specific model instance. """

	def __init__(self, model):
		""" Store specific pre-initialized model instance. """

		self.model = model


	def train_model(self,X,Y):
		""" Train using specific model's training function. """

		self.model.train_model(X,Y)


	def test_model(self,X,Y):
		""" Test using specific model's eval function. """

		labels = []						# List of actual labels
		p_labels = []					# List of model's predictions
		success = 0						# Number of correct predictions
		total_count = 0					# Number of images

		for i in range(len(X)):

			x = X[i]				# Test input
			y = Y[i]				# Actual label
			y_ = self.model.eval(x)		# Model's prediction

			labels.append(y)
			p_labels.append(y_)

			if y == y_:
				success += 1
			total_count +=1



		# Compute Confusion Matrix
		getConfusionMatrixPlot(labels,p_labels,CLASS_LABELS)



if __name__ == "__main__":

	data = sio.loadmat('mat4SVM.mat')
	brain = data['mat4SVM']



	label = np.zeros([202,1])
	label[56:] = 1
	# label = np.zeros([213,1])
	# label[64:] = 1
	X, X_val, Y, Y_val = train_test_split(brain, label, test_size=0.3, random_state=0)


	CLASS_LABELS = ['female','male']


	# Project Data to 200 Dimensions using CCA
	feat_dim = max(X[0].shape)
	projections = Projections(feat_dim,CLASS_LABELS)
	cca_proj,white_cov = projections.cca_projection(X,Y,k=2)

	# project X training on lower dimension
	X = projections.project(cca_proj,white_cov,X)

	# project X_val on lower dimensions
	X_val = projections.project(cca_proj,white_cov,X_val)



	###RUN RIDGE REGRESSION#####
	ridge_m = Ridge_Model(CLASS_LABELS)
	model = Model(ridge_m)

	model.train_model(X,Y)
	model.test_model(X,Y)
	model.test_model(X_val,Y_val)


	####RUN LDA REGRESSION#####

	lda_m = LDA_Model(CLASS_LABELS)
	model = Model(lda_m)

	model.train_model(X,Y)
	model.test_model(X,Y)
	model.test_model(X_val,Y_val)


	####RUN QDA REGRESSION#####

	qda_m = QDA_Model(CLASS_LABELS)
	model = Model(qda_m)

	model.train_model(X,Y)
	model.test_model(X,Y)
	model.test_model(X_val,Y_val)


	####RUN SVM REGRESSION#####

	svm_m = SVM_Model(CLASS_LABELS)
	model = Model(svm_m)

	model.train_model(X,Y)
	model.test_model(X,Y)
	model.test_model(X_val,Y_val)
