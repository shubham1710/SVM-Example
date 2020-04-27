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
clf.fit(a, b)
pred = clf.predict(a)
pred.resize(co,1)
np.savetxt("predicted_test_Y.csv", pred, delimiter=",")


# In[ ]:




