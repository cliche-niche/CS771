import utils
from sklearn import tree,metrics
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from params import *

(X, y_) = utils.loadData( "../../train", dictSize = dictSize )
y_ = y_.astype(int)
y = np.zeros((y_.shape[0], CLASSES + 1))
y[np.arange(y_.size), y_] = 1



clf = tree.DecisionTreeClassifier(min_samples_split=20, )
clf = clf.fit(X[:TRAIN_SET_SIZE], y[:TRAIN_SET_SIZE])


#Predict the response for test dataset
from joblib import dump, load
dump(clf, 'models/DT.joblib')
# Model Accuracy, how often is the classifier correct?


