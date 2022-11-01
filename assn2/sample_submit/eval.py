# from random import shuffle
# from copyreg import pickle
from copy import deepcopy
# from copyreg import pickle
from re import T
from tarfile import XGLTYPE
from webbrowser import get
import utils
import predict
import time as tm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
# This file is intended to demonstrate how we would evaluate your code
# The data loader needs to know how many feature dimensions are there
dictSize = 225
(X, y) = utils.loadData( "train", dictSize = dictSize )
X= X.toarray()
# y=y.toarray()
# pca = decomposition.TruncatedSVD(n_components=8)
# pca.fit(X)
# X = pca.transform(X)
# (Xt, yt) = utils.loadData( "test", dictSize = dictSize )

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
counts=np.bincount(y)
# # maxcount= counts[maxcount]
# X_new=[]
# y_new=[]
# gg=[]
# for i in range(counts.shape[0]):
#     if(counts[i]>1500 ):gg.append(i)
# print(gg)
# gg=[]
# for i in range(counts.shape[0]):
#     if(counts[i]<1500 and counts[i]>300):gg.append(i)
# print(gg)
# gg=[]
# for i in range(counts.shape[0]):
#     if(counts[i]<300 and counts[i]>200):gg.append(i)
# print(gg)
# gg=[]
# for i in range(counts.shape[0]):
#     if(counts[i]<200 and counts[i]>120):gg.append(i)
# print(gg)
# gg=[]
# for i in range(counts.shape[0]):
#     if(counts[i]<120 and counts[i]>50):gg.append(i)
# print(gg)

# gg=[]
# for i in range(counts.shape[0]):
#     if(counts[i]<50):gg.append(i)
# print(gg)
# print('done')

    # for j in range(((maxcount+counts[y[i]]-1)//counts[y[i]])): X_new.append(X[i]), y_new.append(y[i])
# X_new=[]
# y_new=[]

# for i in range(len(X)):
#      if(counts[y[i]]<1500 and counts[y[i]]>300):X_new.append(X[i]), y_new.append(1)
#      if(counts[y[i]]<300 and counts[y[i]]>200):X_new.append(X[i]), y_new.append(1)
#      if(counts[y[i]]<200 and counts[y[i]]>120):X_new.append(X[i]), y_new.append(1)
#      if(counts[y[i]]<120 and counts[y[i]]>50):X_new.append(X[i]), y_new.append(0)
#      if(counts[y[i]]<50):X_new.append(X[i]), y_new.append(0)
#      if(counts[y[i]]>1500):X_new.append(X[i]), y_new.append(1)
# X=np.array(X_new)
# y=np.array(y_new)
le = LabelEncoder()
y = le.fit_transform(y)

counts= np.bincount(y)

# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(counts)

def getPrec (a=y_test, c=5):
    print('here')
    preck = utils.getPrecAtK( a, yPred, c )
    # The macro precision code takes a bit longer to execute due to the for loop over labels
    mpreck = utils.getMPrecAtK( a, yPred, c )

    # According to our definitions, both prec@k and mprec@k should go up as k goes up i.e. for your
    # method, prec@i > prec@j if i > j and mprec@i > mprec@j if i > j. See the assignment description
    # to convince yourself why this must be the case.

    print( "prec@1: %0.3f" % preck[0], "prec@3: %0.3f" % preck[1*(c>1)], "prec@5: %0.3f" % preck[4*(c>1)] )
    # Dont be surprised if mprec is small -- it is hard to do well on rare error classes
    print( "mprec@1: %0.3e" % mpreck[0], "mprec@3: %0.3e" % mpreck[1*(c>1)], "mprec@5: %0.3e" % mpreck[4*(c>1)] )


# # for i in range(X_train.shape[0]):
# #     if(y_train[i]==13):
# #         print(np.nonzero( X_train[i]))
# # print('test')
# # for i in range(X_test.shape[0]):
# #     if(y_test[i]==13):
# #         print(np.nonzero( X_test[i]))

# # maxcount= np.argmax( (np.bincount(y_train)))
# # counts= np.bincount(y_train)
# # maxcount= counts[maxcount]
# # X_new1=[]
# # y_new1=[]

# # X_new2=[]
# # y_new2=[]

# # # X_new3=[]
# # # y_new3=[]

# # # X_new4=[]
# # # y_new4=[]

# # # X_new5=[]
# # # y_new5=[]

# # X_new6=[]
# # y_new6=[]
# XX={}
# X1={}

# for i in range(len(X_train)):
#     if(y_train[i] not in XX):XX[y_train[i]]=[]
#     XX[y_train[i]].append(X_train[i])
#     # if(y_train[i] not in X1):X1[y_train[i]]=[]
# for i in range(counts.shape[0]):
#     XX[i]= np.array(XX[i])
# for i in range(len(X_test)):
#     if(y_test[i] not in X1):X1[y_test[i]]=[]
#     X1[y_test[i]].append(X_test[i])
# np.random.seed(0)
# # classifiers = {}
# # for i in range(counts.shape[0]):
# #     print(i)
# #     for j in range(counts.shape[0]):
# #         if(i>=j):continue
# #         X_new1=[]
# #         y_new1=[]
# #         X_new2=[]
# #         y_new2=[]
# #         # print(len(XX[2]))
# #         if (len(XX[j]) > len(XX[i])): 
# #             # print('here')
# #             for k in range(len(XX[j])// len(XX[i])):
                
# #                 X_new1.extend(XX[i])
# #                 y_new1.extend([0]*len(XX[i]))
      
# #             X_new1.extend(XX[j])
# #             y_new1.extend([1]*len(XX[j]))
# #             for l in range (counts.shape[0]):
# #                 if(l==i or l==j):continue
# #                 al=np.random.randint(0, len(XX[l]), len(XX[j])//40)
# #                 for x in al:
# #                     X_new1.append(XX[l][x])
# #                     y_new1.append(2)
# #                 # y_new1.extend([2]*len(XX[j])//40)

        
# #         else: 
# #             # print('here')

# #             for k in range(len(XX[i])// len(XX[j])):
                
# #                 X_new1.extend(XX[j])
# #                 y_new1.extend([1]*len(XX[j]))
      
# #             X_new1.extend(XX[i])
# #             y_new1.extend([0]*len(XX[i]))

# #             for l in range (counts.shape[0]):
# #                 if(l==i or l==j):continue
# #                 al=np.random.randint(0, len(XX[l]), len(XX[i])//40) 
# #                 for x in al:
# #                     X_new1.append(XX[l][x])
# #                     y_new1.append(2)
                
# #         X_new1=np.array(X_new1)
# #         y_new1=np.array(y_new1)
# #         from sklearn.utils import shuffle
# #         X_new1, y_new1 = shuffle(X_new1, y_new1, random_state=0)

# #         # X_new2.extend(X1[i])
# #         # X_new2.extend(X1[j])
# #         # y_new2.extend([0]*len(X1[i]))
# #         # y_new2.extend([1]*len(X1[j]))
# #         # X_new2=np.array(X_new2)
# #         # y_new2=np.array(y_new2)

# #         # print('here')
# #         clf = XGBClassifier().fit(X_new1, y_new1)
# #         # print (clf.score(X_new2, y_new2))
# #         # cc1=0
# #         # cc0=0
# #         # c1=0
# #         # c0=0
# #         # for l in range(X_new2.shape[0]):
# #         #     if(y_new2[l]==clf.predict([X_new2[l]])[0]):
# #         #         if(y_new2[l]==0):
# #         #             cc0+=1
# #         #         elif(y_new2[l]==1):
# #         #             cc1+=1
# #         #     if(y_new2[l]==0):
# #         #         c0+=1
# #         #     elif(y_new2[l]==1):
# #         #         c1+=1
# #         print(1)

# #         classifiers[(i,j)]=clf
# # # file = open('important', 'wb')
# import pickle
# # # dump information to that file
# # pickle.dump(classifiers, open('classifiersbalancedtripple.pkl', 'wb'))
# # classifiers = {}
# # with (open("myfile", "rb")) as openfile:
# #     while True:
# #         try:
# #             objects.append(pickle.load(openfile))
# #         except EOFError:
# #             break
# classifiers = pickle.load(open('classifiersbalancedtripple.pkl', 'rb'))


# #     if(y_train[i] in [4, 11, 13, 14, 15, 17, 18, 19, 20, 21, 24, 25, 28, 29, 30, 34, 35, 36, 38, 39, 40, 41, 43, 46] ): X_new1.append(X_train[i]), y_new1.append(y_train[i])
    
# # #     elif(y_train[i] in [0, 3] ): X_new2.append(X_train[i]), y_new2.append(y_train[i])
# # #     elif(y_train[i] in [6, 7, 8, 9] ): X_new3.append(X_train[i]), y_new3.append(y_train[i])
# # #     elif(y_train[i] in [10, 11, 12,15,18] ): X_new4.append(X_train[i]), y_new4.append(y_train[i])
# # #     elif(y_train[i] in [4, 14, 19, 22, 24, 25, 29] ): X_new5.append(X_train[i]), y_new5.append(y_train[i])
# #     else :X_new6.append(X_train[i]), y_new6.append(y_train[i])

# # #     # for j in range(((maxcount+counts[y_train[i]]-1)//counts[y_train[i]])): X_new.append(X_train[i]), y_new.append(y_train[i])

# # # XX=[X_new1, X_new2, X_new3, X_new4, X_new5, X_new6  ]

# # # yy=[y_new1, y_new2, y_new3, y_new4, y_new5, y_new6  ]
# # # print(len(y_new1), len(y_new2))
# # from sklearn.utils import shuffle
# # trainf=[]
# # trainr=[]
# # for k in range(len(X_new6)// len(X_new1)) : 
# #     for i in range(len(X_new1)):
# #         trainf.append(X_new1[i])
# #         trainr.append(0)
# # for i in range(len(X_new6)):
# #     trainf.append(X_new6[i])
# #     trainr.append(1)
# # # for j in range (5,6):
# # #     for i in range(len(XX[j])):
# # #         trainf.append(XX[j][i])
# # #         trainr.append(1)
# # shuffle(trainf, trainr, random_state=0)
# # from sklearn.neural_network import MLPClassifier
# # xbfc1 = MLPClassifier()
# # xbfc1.fit(trainf, trainr)
# # m1 = MLPClassifier()
# # m1.fit(X_new1, y_new1)
# # m2 = MLPClassifier()
# # m2.fit(X_new6, y_new6)

# # def bat(a):
# #     if(xbfc1.predict([a])[0]==0):
# #         return m1.predict([a])[0]
# #     else:
# #         return m2.predict([a])[0]

# # # trainf=[]
# # # trainr=[]
# # # for i in range(len(X_new2)):
# # #     trainf.append(X_new2[i])
# # #     trainr.append(0)
# # # for j in range (2,6):
# # #     for i in range(len(XX[j])):
# # #         trainf.append(XX[j][i])
# # #         trainr.append(1)

# # # shuffle(trainf, trainr, random_state=0)
# # # xgbc2 = XGBClassifier()
# # # xgbc2.fit(trainf, trainr)


# # # trainf=[]
# # # trainr=[]
# # # for i in range(len(X_new3)):
# # #     trainf.append(X_new3[i])
# # #     trainr.append(0)
# # # for j in range (3,6):
# # #     for i in range(len(XX[j])):
# # #         trainf.append(XX[j][i])
# # #         trainr.append(1)

# # # shuffle(trainf, trainr, random_state=0)
# # # xgbc3 = XGBClassifier()
# # # xgbc3.fit(trainf, trainr)

# # # trainf=[]
# # # trainr=[]
# # # for i in range(len(X_new4)):
# # #     trainf.append(X_new4[i])
# # #     trainr.append(0)
# # # for j in range (4,6):
# # #     for i in range(len(XX[j])):
# # #         trainf.append(XX[j][i])
# # #         trainr.append(1)

# # # shuffle(trainf, trainr, random_state=0)
# # # xgbc4 = XGBClassifier()
# # # xgbc4.fit(trainf, trainr)

# # # trainf=[]
# # # trainr=[]
# # # for i in range(len(X_new5)):
# # #     trainf.append(X_new5[i])
# # #     trainr.append(0)
# # # for j in range (5,6):
# # #     for i in range(len(XX[j])):
# # #         trainf.append(XX[j][i])
# # #         trainr.append(1)

# # # shuffle(trainf, trainr, random_state=0)
# # # xgbc5 = XGBClassifier()
# # # xgbc5.fit(trainf, trainr)

# # # # X_train, y_train = np.array(X_new), np.array(y_new)
# # # # X_test, y_test = np.array(X_newt), np.array(y_newt)

# # X_new=[]
# # y_new=[]
# # # c=0
# # for i in range(len(X_test)):
# #     if(y_test[i] in [4, 11, 13, 14, 15, 17, 18, 19, 20, 21, 24, 25, 28, 29, 30, 34, 35, 36, 38, 39, 40, 41, 43, 46] ): 
# #         # c+=1
# #         X_new.append(X_test[i]), y_new.append(0)
    
# #     # elif(y_test[i] in [0, 3] ): X_new.append(X_test[i]), y_new.append(1)
# #     # elif(y_test[i] in [6, 7, 8, 9] ): X_new.append(X_test[i]), y_new.append(2)
# #     # elif(y_test[i] in [10, 11, 12,15,18] ): X_new.append(X_test[i]), y_new.append(3)
# #     # elif(y_test[i] in [4, 14, 19, 22, 24, 25, 29] ): X_new.append(X_test[i]), y_new.append(4)
# #     else :X_new.append(X_test[i]), y_new.append(1)
# # # print(c)

# # def battery(a):
# #     if(xbfc1.predict([a])[0]==0):
# #         return 0
# #     else :return 1
# # #     elif(xgbc2.predict([a])[0]==0):
# # #         return 1
# # #     elif(xgbc3.predict([a])[0]==0):
# # #         return 2
# # #     elif(xgbc4.predict([a])[0]==0):
# # #         return 3
# # #     elif(xgbc5.predict([a])[0]==0):
# # #         return 4
# # #     else:
# # #         return 5
# # yPred=[]
# # for x in X_new:
# #     yPred.append(battery(x))

# # yPred=np.reshape(np.array(yPred), (len(yPred), 1))
# # getPrec(np.array(y_new), c=1)

# # for  j in range(2):
# #     countd=0
# #     countf=0
# #     c1=0
# #     c2=0
# #     for i in range(len(X_new)):
# #         if(y_new[i]==j):
# #             countd+= yPred[i]==j
# #             # if(yPred[i]>j):c2+=1
# #             # if(yPred[i]<j):c1+=1
# #             countf+=1
        
# #     if(countf): print(j, countd/countf, countf)#c1/countf, c2/countf, countf)

# # # #######################
# # # print(counts)
# #     # X_train[i]= X_train[i]/np.linalg.norm(X_train[i])


# # from sklearn import tree
# # clf = tree.DecisionTreeClassifier()
# # clf = clf.fit(X_train, y_train)
# # # print( np.mean(clf.predict(X_train)==y_train))
# # # yPred =np.reshape(clf.predict(X_test), (X_test.shape[0],1))
# # # clf.predict_proba(X_test)
# # yPred=[]
# # probs= clf.predict_proba(X_test)
# # for i in range (X_test.shape[0]):
# #    yPred.append( probs[i].argsort()[-5:][::-1])
# #     # np.argpartition(probs[i], -5)[-5:])
# # yPred=np.reshape(np.array(yPred), (len(yPred), 5))
# # getPrec(c=5)

# # # # print('test')
# # # # print('here')
# # # # getPrec()
# # # # print('here1')



# # # from sklearn.svm import SVC
# # # svm_model_linear = SVC( kernel='linear', C = 1).fit(X_train, y_train)
# # # yPred =np.reshape(svm_model_linear.predict(X_test), (X_test.shape[0],1))
# # # print('svm')
# # # getPrec()


# # # clfova = OneVsRestClassifier(SVC()).fit(X_train, y_train)
# # # print( np.mean(clfova.predict(X_train)==y_train))
# # # yPred =np.reshape(clfova.predict(X_test), (X_test.shape[0],1))
# # # print('ova')
# # # getPrec()

# # # # from sklearn.neighbors import KNeighborsClassifier
# # # # knn = KNeighborsClassifier().fit(X_train, y_train)
# # # # yPred =np.reshape(knn.predict(X_test), (X_test.shape[0],1))
# # # # getPrec ()


# # # # from sklearn.naive_bayes import GaussianNB
# # # # gnb = GaussianNB().fit(X_train, y_train)
# # # # # gnb_predictions = gnb.predict(X_test)
# # # # yPred =np.reshape(gnb.predict(X_test), (X_test.shape[0],1))
# # # # getPrec ()
  
# # xgbc = XGBClassifier() #(max_depth=8, n_estimators=1000)
# # xgbc.fit(X_train, y_train)
# # yPred=[]
# # probs= xgbc.predict_proba(X_test)
# # for i in range (X_test.shape[0]):
# #    yPred.append( probs[i].argsort()[-5:][::-1])
# #     # np.argpartition(probs[i], -5)[-5:])
# # yPred=np.reshape(np.array(yPred), (len(yPred), 5))
# # getPrec(c=5)

# # print('xgb')
# # for  j in range(counts.shape[0]):
# #     countd=0
# #     countf=0
# #     for i in range(X_test.shape[0]):
# #         if(y_test[i]==j):
# #             countd+= yPred[i]==j
# #             countf+=1
        
# #     if(countf): print(j,counts[j], countd/countf)
# # # getPrec ()
# # # # ypred = xgbc.predict(X_train)

# # # print( np.mean(ypred==y_train))


from sklearn.neural_network import MLPClassifier
# # clfnn = MLPClassifier()
# # clfnn.fit(X_train, y_train)
# # pickle.dump(clfnn, open('nn.pkl', 'wb'))
# clfnn= pickle.load(open('nn.pkl', 'rb'))
# yPred=[]
# probs= clfnn.predict_proba(X_test)
# # for x in X_test:
# #     if (xbfc1.predict([x])[0]==0): yPred.append(m1.predict_proba([x])[0].argsort()[-5:][::-1])
# #     else : yPred.append(m2.predict_proba([x])[0].argsort()[-5:][::-1])

# for i in range (X_test.shape[0]):
#     # print(i)
#     a1=probs[i].argsort()[-5:][::-1]

#     # for k in range (1):
#     #     for j in range(4,0,-1):
#     #         a0= [min(a1[j-1], a1[j]), max(a1[j-1], a1[j]), 100]
#     #         if((a0[classifiers[(min(a1[j-1], a1[j]), max(a1[j-1], a1[j]))].predict([X_test[i]])[0]] ==a1[j]) and ((max(classifiers[(min(a1[j-1], a1[j]), max(a1[j-1], a1[j]))].predict_proba([X_test[i]])[0])>0.65 ) or (max(classifiers[(min(a1[j-1], a1[j]), max(a1[j-1], a1[j]))].predict_proba([X_test[i]])[0])>0.45 and (probs[i][a1[j-1]]- probs[i][a1[j]])/probs[i][a1[j]]<2))):
#     #             a1[j-1], a1[j]= a1[j], a1[j-1]
#     #         else:
#     #             if(y_test[i]==a1[j] and (a0[classifiers[(min(a1[j-1], a1[j]), max(a1[j-1], a1[j]))].predict([X_test[i]])[0]] ==a1[j])): print ( j, probs[i][a1[j-1]], probs[i][a1[j]], classifiers[(min(a1[j-1], a1[j]), max(a1[j-1], a1[j]))].predict_proba([X_test[i]])[0])
                
#     # if(a)

        
#     yPred.append(a1)
#     # np.argpartition(probs[i], -5)[-5:])

# yPred=np.reshape(np.array(yPred), (len(yPred),5))
# getPrec(c=5)

# print('nn')
# # fuk=[]
# # for  j in range(counts.shape[0]):
# #     countd=0
# #     countf=0
# #     for i in range(X_test.shape[0]):
# #         if(y_test[i]==j):
# #             countd+= yPred[i]==j
# #             countf+=1
        
# #     if(countf): print(j,counts[j], countd/countf)
# #     if(countd[0]/countf<0.75) :fuk.append(j)
# #     # if(countd/countf <0.75): print('not gg', j, counts[j])
# # print(fuk)
# # getPrec ()


# # # from sklearn.neighbors import NearestCentroid
# # # clfnc = NearestCentroid()
# # # clfnc.fit(X_train, y_train)
# # # yPred =np.reshape(clfnc.predict(X_test), (X_test.shape[0],1))
# # # getPrec ()


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.utils import class_weight
# classes_weights = class_weight.compute_sample_weight(
#     class_weight='balanced',
#     y=y_train
# )

# # clfmax = XGBClassifier()
# ynew=[]
# yt=[]
# for x in range(X_train.shape[0]):
#     if(y_train[x]!=1 and y_train[x]!=2 ): ynew.append(0)
#     else: ynew.append(y_train[x])

# for x in range(X_test.shape[0]):
#     if(y_test[x]!=1 and y_test[x]!=2 ): yt.append(0)
#     else: yt.append(y_test[x])

   
# classes_weights2 = class_weight.compute_sample_weight(
#     class_weight='balanced',
#     y=ynew
# )
# clfrfc = XGBClassifier()
# clfrfc2= XGBClassifier()
# clfnn = MLPClassifier()

# clfrfc.fit(X_train, y_train, sample_weight=classes_weights)
# clfrfc2.fit(X_train, ynew, sample_weight=classes_weights2)
# clfnn.fit(X_train, y_train)

import pickle
# pickle.dump(clfrfc, open('rfc.pkl', 'wb'))
# pickle.dump(clfrfc2, open('rfc2.pkl', 'wb'))
# pickle.dump(clfnn, open('nn.pkl', 'wb'))

clfrfc= pickle.load(open('rfc.pkl', 'rb'))
clfrfc2= pickle.load(open('rfc2.pkl', 'rb'))
clfnn= pickle.load(open('nn.pkl', 'rb'))


probs= clfrfc.predict_proba(X_test)
probsnn= clfnn.predict_proba(X_test)
probs2= clfrfc2.predict_proba(X_test)
# probsmax2= clfmax2.predict_proba(X_test)
# predsmax= clfmax.predict(X_test)
# predsmax2= clfmax2.predict(X_test)

# probmaxf=[]
# for i in range (X_test.shape[0]):
#     a1=(probs[i] + probsnn[i])/2
#     # else:
#     a2= [0, probsmax[i][1],probsmax[i][2],0] + [0]* (43)
#     a2+= (probs[i] + probsnn[i])/2
#     a2[1]-=a1[1]
#     a2[2]-=a1[2]
#     probmaxf.append(a2)
# probsmaxf=np.array(probmaxf)
# print(clfrfc2.score (X_test, yt))
# print(X_test.shape)
# print(np.sum(clfrfc2.predict(X_test)==yt))
# for i in range (X_test.shape[0]):
#     if(clfrfc2.predict(X_test)[i]==yt[i]):
#         print(i, clfrfc2.predict(X_test)[i], yt[i], probs2[i])
yPred=[]
yPred1=[]
yPred2=[]
yPred3=[]
yPred4=[]
all = deepcopy(probs)
for i in range (X_test.shape[0]):
    al= probs[i].argsort()[-5:][::-1]

    a2= probsnn[i].argsort()[-5:][::-1]
    a3= probs2[i].argsort()[-5:][::-1]
    yPred1.append(al)
    yPred2.append(a2)
    # yPred3.append(a3)
    # print(probsmax.shape)
    # probsmax[i]= [0, probsmax[i][1], probsmax[i][2]] + [0]*(probsmax.shape[1]-3)
    # all= [,probs2[i][1], probs2[i][2],0] + [0]* (43)
    probs[i]= probs[i]*0.7+probsnn[i]*0.3
    
    al= probs[i].argsort()[-5:][::-1]
    for x in al:
        probs[i][x]= (probs[i][x]*0.5+probsnn[i][x]*0.2)*3
        # print(probs2[i].shape)
    # all[i]= probs[i]*0.5+probsnn[i]*0.5

    # all= deepcopy(probs[i])
    
    
    # for x in range(probs[i].shape[0]):
    #     probs[i][x]= probs[i][x]**2+ probsnn[i][x]**2


        # if(len(al)==5): break
    yPred.append( probs[i].argsort()[-5:][::-1])
    # probs[i]= probs[i]+  all[i]*0.5 + probsnn[2]*0.2

    all[i][1]=probs2[i][1]
    all[i][2]= probs2[i][2]

    probs[i] = probs[i] + all[i]*0.1
    yPred4.append( probs[i].argsort()[-5:][::-1])

    # np.argpartition(probs[i], -5)[-5:])
yPred=np.reshape(np.array(yPred), (len(yPred), 5))
getPrec(c=5)
yPred=np.reshape(np.array(yPred1), (len(yPred1), 5))
getPrec(c=5)
yPred=np.reshape(np.array(yPred2), (len(yPred2), 5))
getPrec(c=5)
# yPred=np.reshape(np.array(yPred3), (len(yPred3), 5))
# getPrec(c=5)
yPred=np.reshape(np.array(yPred4), (len(yPred4), 5))
getPrec(c=5)

# # yPred =np.reshape(clfrfc.predict(X_test), (X_test.shape[0],1))
# # print('rfc')
# # getPrec ()

# # # from sklearn.ensemble import GradientBoostingClassifier
# # clfgbc = XGBClassifier()
# # clfgbc.fit(X_train, y_train)
# # yPred =np.reshape(clfgbc.predict(X_test), (X_test.shape[0],1))
# # print('gb')
# # yPred=[]
# # probs= clfgbc.predict_proba(X_test)
# # for i in range (X_test.shape[0]):
# #    yPred.append( probs[i].argsort()[-5:][::-1])
# #     # np.argpartition(probs[i], -5)[-5:])
# # yPred=np.reshape(np.array(yPred), (len(yPred), 5))
# # getPrec(c=5)

# # getPrec ()

# # print (clf.score(X, y))
# # print(X.shape)
# # print(y.shape)
# # print(clf.predict(X).shape)
# #######################

# # # Get error class predictions from predict.py and time the thing
# # tic = tm.perf_counter()
# # # print(clf.predict_proba(X))
# # yPred =np.reshape(clf.predict(X_test), (X_test.shape[0],1))
# # yPred =np.reshape(xgbc.predict(X_test), (X_test.shape[0],1))
# # yPred =np.reshape(clfova.predict(X_test), (X_test.shape[0],1))
# # yPred =np.reshape(knn.predict(X_test), (X_test.shape[0],1))
# # yPred =np.reshape(gnb.predict(X_test), (X_test.shape[0],1))
# # yPred =np.reshape(svm_model_linear.predict(X_test), (X_test.shape[0],1))
# # yPred =np.reshape(clfnn.predict(X_test), (X_test.shape[0],1))

# # yPred =np.reshape(clfnc.predict(X_test), (X_test.shape[0],1))
# # yPred =np.reshape(clfrfc.predict(X_test), (X_test.shape[0],1))
# # # toc = tm.perf_counter()

# # yPred =np.reshape(clfgbc.predict(X_test), (X_test.shape[0],1))
# # toc = tm.perf_counter()

# # print( "Total time taken is %.6f seconds " % (toc - tic) )


# from sklearn.utils import class_weight

# # classifiers={}
# import pickle
# # pickle.dump(classifiers, open('classifiers1v1.pkl', 'wb'))
# classifiers= pickle.load(open('classifiers1v1.pkl', 'rb'))

# for i in range (counts.shape[0]):
#     ynew = list(deepcopy(y_train))
#     for x in range (len(ynew)):
#         if(ynew[x]!=i):
#             ynew[x]= 0
#         else:
#             ynew[x]= 1
#     ynew= np.array(ynew)
#     # print(classifiers[i].score (X_test, ynew), (np.sum(ynew)),(np.sum(classifiers[i].predict(X_test))), np.sum(classifiers[i].predict(X_test)==ynew))
#     # print(ynew)
#     classes_weights = class_weight.compute_sample_weight(
#     class_weight='balanced',
#     y=ynew
#     )
#     xx= MLPClassifier()
#     xx.fit(X_train, ynew)

#     classifiers[i]= xx
    
#     ynew = list(deepcopy(y_test))
#     for x in range (len(ynew)):
#         if(ynew[x]!=i):
#             ynew[x]= 0
#         else:
#             ynew[x]= 1
#     ynew= np.array(ynew)
#     print(classifiers[i].score (X_test, ynew), (np.sum(ynew)),(np.sum(classifiers[i].predict(X_test))), np.sum(classifiers[i].predict(X_test)==ynew))
#     # print(i)
# import pickle
# pickle.dump(classifiers, open('classifiers1v1vn.pkl', 'wb'))
# # classifiers= pickle.load(open('classifiers1v1.pkl', 'rb'))

# def getProb(a):
#     prob= np.zeros(counts.shape[0])
#     for i in range (counts.shape[0]):
#         prob[i]= classifiers[i].predict_proba([a])[0][1]
#     return prob
# yPred=[]
# for i in range (X_test.shape[0]):

# # yPred4.append( probs[i].argsort()[-5:][::-1])
#     yPred.append(getProb(X_test[i]).argsort()[-5:][::-1])
#     # print(i)
# yPred=np.reshape(np.array(yPred), (len(yPred), 5))
# getPrec(c=5)

