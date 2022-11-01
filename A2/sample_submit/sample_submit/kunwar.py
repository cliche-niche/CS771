from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import utils
from sklearn.neural_network import MLPClassifier
from params import *
from xgboost import XGBClassifier
from kunwar_utils import *

(X, y) = utils.loadData( "../../train", dictSize = dictSize )
X= X.toarray()
X_train, X_test, y_train, y_test, le = train_test_split_encode(X, y, test_size=0.2, random_state=42)
ynew = generate_y(y_train)
yt = generate_y(y_test)
from sklearn.utils import class_weight
classes_weights = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=y_train
)
classes_weights2 = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=ynew
)

clfrfc = XGBClassifier()
clfrfc2= XGBClassifier()
clfnn = MLPClassifier()

clfrfc.fit(X_train, y_train, sample_weight=classes_weights)
clfrfc2.fit(X_train, ynew, sample_weight=classes_weights2)
clfnn.fit(X_train, y_train)

import pickle
pickle.dump(X_test, open("X_test.pkl", "wb"))
pickle.dump(y_test, open("y_test.pkl", "wb"))
pickle.dump(le, open('le.pkl', 'wb'))
pickle.dump(clfrfc, open('rfc.pkl', 'wb'))
pickle.dump(clfrfc2, open('rfc2.pkl', 'wb'))
pickle.dump(clfnn, open('nn.pkl', 'wb'))