import cv2
import IPython
from numpy.random import uniform
import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys


from projection_brain import Project2D, Projections
import scipy.io as sio
from sklearn.model_selection import train_test_split


from confusion_mat import getConfusionMatrix
from confusion_mat import plotConfusionMatrix

from ridge_mat4SVM import Ridge_Model
from qda_mat4SVM import QDA_Model
from lda_mat4SVM import LDA_Model
from svm_mat4SVM import SVM_Model


CLASS_LABELS = ['female', 'male']


def eval_model(X,Y,k,model_key,proj):
	# PROJECT DATA
	cca_proj,white_cov = proj.cca_projection(X,Y,k=k)

	X_p = proj.project(cca_proj,white_cov,X)
	X_val_p = proj.project(cca_proj,white_cov,X_val)

	# TRAIN MODEL
	model = models[model_key]

	model.train_model(X_p,Y)
	acc,cm = model.test_model(X_val_p,Y_val)
	# print('here is acc and cm', acc, cm)
	return acc,cm


class Model():
	""" Generic wrapper for specific model instance. """


	def __init__(self,model):
		""" Store specific pre-initialized model instance. """

		self.model = model


	def train_model(self,X,Y):
		""" Train using specific model's training function. """

		self.model.train_model(X,Y)



	def test_model(self,X,Y):
		""" Test using specific model's eval function. """

		labels = []						# List of actual labels
		p_labels = []					# List of model's predictions
		success = 0.0					# Number of correct predictions
		total_count = 0.0				# Number of images


		for i in range(len(X)):

			x = X[i]					# Test input
			y = Y[i]					# Actual label
			y_ = self.model.eval(x)		# Model's prediction
			labels.append(y)
			p_labels.append(y_)

			if y == y_:
				success += 1.0
			total_count +=1.0


		return success/total_count, getConfusionMatrix(labels,p_labels)



if __name__ == "__main__":

	data = sio.loadmat('mat4SVM_Af_Am_volume_TCTV_age.mat')
	brain = data['mat4SVM']
	class_label= sio.loadmat('class_Af_Am.mat')


	# label = np.zeros([202,1])
	# label[56:] = 1
	label = np.zeros([213,1])
	label[64:] = 1
	X, X_val, Y, Y_val = train_test_split(brain, label, test_size=0.3, random_state=0)


	# Project Data to 200 Dimensions using CCA
	feat_dim = max(X[0].shape)
	projections = Projections(feat_dim,CLASS_LABELS)



	models = {}						# Dictionary of key: model names, value: model instance

	#########MODELS TO EVALUATE############
	# qda_m = QDA_Model(CLASS_LABELS)
	# models['qda'] =  Model(qda_m)
	# print('finishing QDA model')
	# print('staring ridge model ')
	# lda_m = LDA_Model(CLASS_LABELS)
	# models['lda'] = Model(lda_m)
	# print('finishing LDA model')
	# print('starting ridge model')
	#
	# ridge_m = Ridge_Model(CLASS_LABELS)
	# models['ridge'] = Model(ridge_m)
	# #
	# ridge_m_10 = Ridge_Model(CLASS_LABELS)
	# ridge_m_10.lmda = 0.05
	# models['ridge_lmda_0.05'] = Model(ridge_m_10)
	# # #
	# ridge_m_01 = Ridge_Model(CLASS_LABELS)
	# ridge_m_01.lmda = 3.5
	# models['ridge_lmda_8'] = Model(ridge_m_01)
	# #
	# svm_m = SVM_Model(CLASS_LABELS)
	# models['svm'] = Model(svm_m)
	#
	# svm_m_10 = SVM_Model(CLASS_LABELS)
	# svm_m_10.C = 10
	# models['svm_C_2'] = Model(svm_m_10)
	#
	# svm_m_01 = SVM_Model(CLASS_LABELS)
	# svm_m_01.C = 7
	# models['svm_C_007'] = Model(svm_m_01)




	#########GRID SEARCH OVER MODELS############
	highest_accuracy = 0			# Highest validation accuracy
	best_model_name = None			# Best model name
	best_model = None				# Best model instance

	# K = [2,10,40,70,100,120,150,180,200,240]# List of dimensions
	K = [1,2,3,4,5,10,15,20,25,30,35,40,45]
	#

	# # print(models.keys())
	# for model_key in models.keys():
	# 	val_acc = []				# List of model's accuracies for each dimension
	# 	for k in K:
	#
	# 		# Evaluate specific model's validation accuracy on specific dimension
	# 		acc,c_m = eval_model(X,Y,k,model_key,projections)
	#
	# 		val_acc.append(acc)
	#
	# 		if acc > highest_accuracy:
	# 			highest_accuracy = acc
	# 			best_model_name = model_key
	# 			best_cm = c_m
	# 	print('here is best acc', val_acc)
	# 	print('here is best_cm', best_cm)
	# 	print('here is best model name', best_model_name)
	# 	# Plot specific model's accuracies across validation error
	# 	plt.plot(K,val_acc,label=model_key)
	#
	#
	# # Display aggregate plot of models across validation error
	# plt.legend()
	# plt.xlabel('Dimension')
	# plt.ylabel('Accuracy')
	# plt.show()
	#
	#
	# # Plot best model's confusion matrix
	# plotConfusionMatrix(best_cm,CLASS_LABELS)



# cross validation- using 4-fold CV FOR RIDGE REGRESSIONv
# lamda = [0,0.05, 0.1, 0.15, 0.2, 0.4, 1, 1.4]
ind = list(range(np.shape(brain)[0]))
random.shuffle(ind)
brain = brain[ind]
label = label[ind]
lamda = [0.1,0.5,0.7,1,2,5,8,12]
Kc = 4;
Ntotal = np.shape(brain)[0]
Ns = int(np.shape(brain)[0]*((Kc-1)/Kc))
print('here is Ns',  Ns)
# number of features
n = np.shape(brain)[1]
Nv = int(Ns/(Kc-1))	  # should be 1/4 of the data

Etrain = np.zeros((1,8))
Evalid = np.zeros((1,8))
Ecv = np.zeros((4,1))
Ecvt = np.zeros((4,1))
highest_accuracy = 0
store_cm = {}
for l in range(len(lamda)):
	ridge_cv = LDA_Model(CLASS_LABELS)
	# ridge_cv.C = lamda[l]
	# models['ridge_cv'] = Model(ridge_cv)
	for c in range(4):
		xv = brain[c*Nv:(c+1)*Nv]
		yv = label[c*Nv: (c+1)*Nv]
		x = np.delete(brain,list(range(c*Nv, (c+1)*Nv)),0)
		y = np.delete(label,list(range(c*Nv, (c+1)*Nv)),0)

		model = Model(ridge_cv)
		k = 2
		cca_proj,white_cov = projections.cca_projection(x,y,k=k)

		X_p = projections.project(cca_proj,white_cov,x)
		X_val_p = projections.project(cca_proj,white_cov,xv)

		# TRAIN MODEL
		# model = models[model_key]

		model.train_model(X_p,y)
		acc,cm = model.test_model(X_val_p,yv)

		if acc>highest_accuracy:
			best_cm = cm
			highest_accuracy = acc

		Ecv[c] = acc

	store_cm[str(l)] = best_cm
	plotConfusionMatrix(best_cm,CLASS_LABELS)
	Evalid[0,l] = np.mean(Ecv)

print('here is Evalid', Evalid)
