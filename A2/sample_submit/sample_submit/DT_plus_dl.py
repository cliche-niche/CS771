import utils
from sklearn import tree,metrics
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from params import *

(X, y_) = utils.loadData( "../../train", dictSize = dictSize )
y = y_.astype(int)


# Assign 1 if classes in DT_1_CLASSES else 0
# for i in range(len(y)):
#     if y[i] not in DT_1_CLASSES:
#         y[i] = 0
#     else:
#         y[i] = 1




X_train = X[:TRAIN_SET_SIZE]
y_train = y[:TRAIN_SET_SIZE]
X_test = X[TRAIN_SET_SIZE:]
y_test = y[TRAIN_SET_SIZE:]
 

#  # Create Decision Tree classifer object
# clf = tree.DecisionTreeClassifier(min_samples_split=15, )
# # Train Decision Tree Classifer
# clf = clf.fit(X_train,y_train)
# #Predict the response for test dataset
# y_pred = clf.predict(X_test)
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='sgd',learning_rate = "adaptive",learning_rate_init= 0.1, hidden_layer_sizes=(128,64,32,16,16,16), random_state=1, max_iter=1000, verbose= True)
# clf.fit(X_train, y_train)
# # Predict the response for test dataset
# y_pred = clf.predict(X_test)
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# # print incorrect predictions
# for i in range(len(y_pred)):
#     if y_pred[i] != y_test[i]:
#         print(clf.predict_proba(X_test[i,:].reshape(1,-1)))




  
