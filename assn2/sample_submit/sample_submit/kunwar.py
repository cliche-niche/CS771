from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import utils
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from kunwar_utils import *

dictSize = 225
(X, y) = utils.loadData( "../../train", dictSize = dictSize )
X= X.toarray()
X_train, X_test, y_train, y_test, le = train_test_split_encode(X, y, test_size=0.0, random_state=42)
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
clfrfc2= MLPClassifier()
clfnn = MLPClassifier()

clfrfc.fit(X_train, y_train, sample_weight=classes_weights)
clfrfc2.fit(X_train, ynew)
clfnn.fit(X_train, y_train)

import pickle
pickle.dump(X_test, open("models/X_test.pkl", "wb"))
pickle.dump(y_test, open("models/y_test.pkl", "wb"))
pickle.dump(le, open('models/le.pkl', 'wb'))
pickle.dump(clfrfc, open('models/rfc.pkl', 'wb'))
pickle.dump(clfrfc2, open('models/rfc2.pkl', 'wb'))
pickle.dump(clfnn, open('models/nn.pkl', 'wb'))