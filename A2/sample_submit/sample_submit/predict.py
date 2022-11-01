from copy import deepcopy
import numpy as np
from numpy import random as rand
from joblib import load
from params import *
import tensorflow as tf
import pickle
# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# PLEASE BE CAREFUL THAT ERROR CLASS NUMBERS START FROM 1 AND NOT 0. THUS, THE FIFTY ERROR CLASSES ARE
# NUMBERED AS 1 2 ... 50 AND NOT THE USUAL 0 1 ... 49. PLEASE ALSO NOTE THAT ERROR CLASSES 33, 36, 38
# NEVER APPEAR IN THE TRAINING SET NOR WILL THEY EVER APPEAR IN THE SECRET TEST SET (THEY ARE TOO RARE)

# Input Convention
# X: n x d matrix in csr_matrix format containing d-dim (sparse) bag-of-words features for n test data points
# k: the number of compiler error class guesses to be returned for each test data point in ranked order

# Output Convention
# The method must return an n x k numpy nd-array (not numpy matrix or scipy matrix) of classes with the i-th row 
# containing k error classes which it thinks are most likely to be the correct error class for the i-th test point.
# Class numbers must be returned in ranked order i.e. the label yPred[i][0] must be the best guess for the error class
# for the i-th data point followed by yPred[i][1] and so on.

# CAUTION: Make sure that you return (yPred below) an n x k numpy nd-array and not a numpy/scipy/sparse matrix
# Thus, the returned matrix will always be a dense matrix. The evaluation code may misbehave and give unexpected
# results if an nd-array is not returned. Please be careful that classes are numbered from 1 to 50 and not 0 to 49.

def get_top_k_preds(model_code,model,x,k):
	"""Directs the call to the appropriate predictor function using the model_code. 
	Add your predictor function in this file
	Then modify this function
	Add the model_code to params.py"""

	if(model_code == DT_TREE_CODE):
		return DT_preds(model,x,k)
	if(model_code ==  DL_CODE):
		return DL_preds(model,x,k)
	if(model_code == KUNWAR_CODE):
		return Kunwar_preds(model,x,k)

def Kunwar_preds(model,x,k, optional = None):
    
	clfrfc= model[0]
	clfrfc2= model[1]
	clfnn= model[2]
	probs= clfrfc.predict_proba([x])
	probsnn= clfnn.predict_proba([x])
	probs2= clfrfc2.predict_proba([x])

	# Convert arrays to 1d arrays
	probs= probs[0]
	probsnn= probsnn[0]
	probs2= probs2[0]

	all = deepcopy(probs)
	al= probs.argsort()[-k:][::-1]
	probs= probs*0.7+probsnn*0.3
	al= probs.argsort()[-k:][::-1]
	# al = np.reshape(al,(al.shape[1]))
	# print(al.shape)
	for j in al:
		probs[j]= (probs[j]*0.5+probsnn[j]*0.2)*3
	yPred = probs.argsort()[-k:][::-1]
	# all[1]=probs2[1]
	# all[2]= probs2[2]
	# # print(all.shape, probs.shape, probsnn.shape, probs2.shape)
	# probs = probs + all*0.1
	yPred=np.reshape(np.array(yPred), k)
	# print(yPred)
	le = model[3]
	yPred = le.inverse_transform(yPred)
	# print(yPred)
	return yPred
def DL_preds(model,x,k):
	"""Takes a DL model (tf.keras.models.Sequential). Predicts the top k classes """
	# Divide x by its norm
	x = x.toarray()
	norm = np.linalg.norm(x)
	x = x/norm
	preds = model(x)
	top_k = tf.math.top_k(preds,k)[-1]
	return np.reshape(top_k,(top_k.shape[1]))	

def DT_preds(model,x,k):
	"""Takes a DT model (sklearn.tree.DecisionTreeClassfier). Predicts the top k classes """
	y_pred = model.predict_proba(x)
	probs = [-1]
	for i in range(1,CLASSES + 1):
		probs.append(1.0 - y_pred[i][0][0])
	res = (sorted(range(len(probs)), key = lambda sub: probs[sub])[-k:])
	res.reverse()
	return np.array(res)

def findErrorClass( X, k ):
	# Find out how many data points we have
	n = X.shape[0]
	print(n)
	# Load and unpack a dummy model to see an example of how to make predictions
	# The dummy model simply stores the error classes in decreasing order of their popularity
	model = [pickle.load(open('rfc.pkl', 'rb')),pickle.load(open('rfc2.pkl', 'rb')),pickle.load(open('nn.pkl', 'rb')), pickle.load(open('le.pkl', 'rb'))]
	y_pred = []
	for j in range(n):
		features = np.reshape(X[j], (1, X[j].shape[0]))
		y_pred.append(get_top_k_preds(KUNWAR_CODE,model, features,k))
	return np.array(y_pred)