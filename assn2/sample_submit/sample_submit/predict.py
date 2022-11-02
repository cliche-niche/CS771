from copy import deepcopy
import numpy as np
from numpy import random as rand
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


def findErrorClass( X, k ):
	# Find out how many data points we have
	n = X.shape[0]
	# Load and unpack a dummy model to see an example of how to make predictions
	# The dummy model simply stores the error classes in decreasing order of their popularity
	model = [pickle.load(open('models/rfc.pkl', 'rb')),pickle.load(open('models/rfc2.pkl', 'rb')),pickle.load(open('models/nn.pkl', 'rb')), pickle.load(open('models/le.pkl', 'rb'))]
	# npzModel = np.load( "model.npz" )
	# model = npzModel[npzModel.files[0]]
	# Let us predict a random subset of the 2k most popular labels no matter what the test point
	# shortList = model[0:2*k]
	clfrfc= model[0]
	clfrfc2= model[1]
	clfnn= model[2]
	probs= clfrfc.predict_proba(X)
	probsnn= clfnn.predict_proba(X)
	probs2= clfrfc2.predict_proba(X)
	all = deepcopy(probs)
	all[:,1]=probs2[:,1]
	all[:,2]= probs2[:,2]
	probs= probs*0.7+probsnn*0.3
	



	# Make sure we are returning a numpy nd-array and not a numpy matrix or a scipy sparse matrix
	yPred = np.zeros( (n, k) )
	for i in range( n ):

		al= probs[i].argsort()[-k:][::-1]
		for j in al:
			probs[j]= (probs[j]*0.5+probsnn[j]*0.2)*3
		
		probs[i] = probs[i] + all[i]*0.1
		al= probs[i].argsort()[-k:][::-1]
		al=np.reshape(np.array(al), k)
		# print(yPred)
		le = model[3]
		# print (yPred)
		yPred[i] = le.inverse_transform(al)
	return yPred