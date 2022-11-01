import pickle
from copy import deepcopy
from sample_submit.sample_submit.kunwar import X_train
from kunwar_utils import *

clfrfc= pickle.load(open('rfc.pkl', 'rb'))
clfrfc2= pickle.load(open('rfc2.pkl', 'rb'))
clfnn= pickle.load(open('nn.pkl', 'rb'))

probs= clfrfc.predict_proba(X_test)
probsnn= clfnn.predict_proba(X_test)
probs2= clfrfc2.predict_proba(X_test)

yPred=[]
all = deepcopy(probs)
for i in range (X_test.shape[0]):
    al= probs[i].argsort()[-5:][::-1]

    a2= probsnn[i].argsort()[-5:][::-1]
    a3= probs2[i].argsort()[-5:][::-1]
    probs[i]= probs[i]*0.7+probsnn[i]*0.3
    
    al= probs[i].argsort()[-5:][::-1]
    for x in al:
        probs[i][x]= (probs[i][x]*0.5+probsnn[i][x]*0.2)*3
    yPred.append( probs[i].argsort()[-5:][::-1])
    all[i][1]=probs2[i][1]
    all[i][2]= probs2[i][2]

    probs[i] = probs[i] + all[i]*0.1

yPred=np.reshape(np.array(yPred), (len(yPred), 5))
getPrec(c=5)
