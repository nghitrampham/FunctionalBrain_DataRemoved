import numpy as np
from utils import create_one_hot_label
from utils import subtract_mean_from_data
from utils import compute_covariance_matrix

import matplotlib.pyplot as plt
import sys
from numpy.linalg import inv
from numpy.linalg import det
from sklearn.svm import LinearSVC
from projection_brain import Project2D, Projections
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from confusion_mat import getConfusionMatrixPlot
import scipy.io as sio

data = sio.loadmat('mat4SVM_Af_Am_volume_TCTV_age.mat')
brain = data['mat4SVM']
class_label= sio.loadmat('class_Af_Am.mat')

label = np.zeros([213,1])
label[64:] = 1
X_train, X_val, Y_train, Y_val = train_test_split(brain, label, test_size=0.3, random_state=0)
CLASS_LABELS = ['female','male']



def class_prior(X,Y):
    count_1 = 0
    prior = []
    size = np.shape(X)[0]
    print('here is size', size)
    for i in range(size):
        if Y[i] == 0:
            count_1 = count_1+1
    prior.append(count_1/size)
    prior.append((size-count_1)/size)

    return prior

def train_model(X,Y):

    holder_1 = []
    holder_2 = []
    for i in range(len(X)):
        if Y[i] == 0:
            holder_1.append(X[i])
        elif Y[i] ==1:
            holder_2.append(X[i])

    state_dim = np.shape(X)[1];
    mean_1 = ((1/len(holder_1))*sum(holder_1)).reshape(1, state_dim)
    # print('here is shape of X', state_dim )
    # mean_1 = mean_1.reshape(1,state_dim )
    #make sure to reshape it, numpy array is weird
    mean_2 = (1/len(holder_2))*sum(holder_2).reshape(1,state_dim )

    cov_1 = np.zeros([np.shape(X)[1],np.shape(X)[1]])
    cov_2 = np.zeros([state_dim,state_dim])
    for k in range(len(X)):
        if Y[k]==0:
            tmp = ((X[k] - mean_1[0]).T).dot(X[k]-mean_1[0])
            cov_1 = np.add(cov_1,tmp)
        elif Y[k] ==1:
            tmp = ((X[k] - mean_2[0]).T).dot(X[k]-mean_2[0])
            cov_2 = np.add(cov_2,tmp)

            # print('here is size of cov_1 and cov_2', np.shape(cov_1), np.shape(cov_2))

    cov_class1= (cov_1/len(holder_1)).reshape(state_dim ,state_dim )
    cov_class2 = (cov_2/len(holder_2)).reshape(state_dim ,state_dim )
    # print('here is size of cov_class1', np.shape(cov_class1))
    # print('here is size of cov_class2', np.shape(cov_class2))
    # print('here is size of mean1', np.shape(mean_1))
    return mean_1, mean_2, cov_class1, cov_class2


prob_class = class_prior(X_train, Y_train)
mean1, mean2, cov1, cov2 = train_model(X_train,Y_train)
Sigma = prob_class[0]*cov1 + prob_class[1]*cov2
canonical = (mean1 - mean2).dot(inv(Sigma + 0.001*np.eye(Sigma.shape[0])))
# threshold = np.log(prob_class[0]/prob_class[1])
threshold = np.log(prob_class[0]/prob_class[1])
def test_model(X,Y, threshold):
    """ Test using specific model's eval function. """

    labels = []						# List of actual labels
    p_labels = []					# List of model's predictions
    success = 0						# Number of correct predictions
    total_count = 0					# Number of images

    for i in range(len(X)):

        x = X[i]				# Test input
        y = Y[i]				# Actual label
        x = x.reshape(len(x),1)
        projected = canonical.dot(x)	# Model's prediction
        # print('here is canonial', np.shape(canonical))
        # print('here is x', np.shape(x))
        # print('here is projected', np.shape(projected))
        if projected > threshold:
            y_ = 0
        else:
            y_ = 1

        labels.append(y)
        p_labels.append(y_)

        if y == y_:
            success += 1
            total_count +=1


    # return success/total_count, getConfusionMatrix(labels,p_labels)
    # Compute Confusion Matrix
    getConfusionMatrixPlot(labels,p_labels,CLASS_LABELS)

test_model(X_train, Y_train, threshold)
test_model(X_val, Y_val,threshold)
