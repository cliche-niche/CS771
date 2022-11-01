from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import utils
from sklearn.neural_network import MLPClassifier
from params import *
from xgboost import XGBClassifier


def train_test_split_encode(X, y, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return X_train, X_test, y_train, y_test, le

def generate_y(x, conserved_index_list = [1,2]):
    y = []
    for i in range(x.shape[0]):
        if(x[i] not in conserved_index_list ): y.append(0)
        else: y.append(x[i])
    return y
