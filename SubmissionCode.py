#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn import metrics
import warnings
a=np.genfromtxt('train_X.csv',delimiter=',')
b=np.genfromtxt('train_Y.csv',delimiter=',')
a=np.delete(a,0,0)
co=np.shape(a)[0]
clf = SVC(kernel='linear')
x_train,x_test,y_train,y_test=train_test_split(a,b,test_size=0.3)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
coo=np.shape(x_test)[0]
pred.resize(coo,1)
np.savetxt("predicted_test_Y.csv", pred, delimiter=",")


# In[ ]:




