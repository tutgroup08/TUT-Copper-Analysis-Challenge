# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 01:35:49 2016

@author: zoey
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 00:54:47 2016

@author: zoey
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 10:20:45 2016

@author: zoey
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:09:20 2016

@author: Lingyu
"""
import numpy as np
import csv
#from sklearn import metrics
from sklearn import svm
#from sklearn.cross_validation import train_test_split
#from sklearn import cross_validation
#from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#import math
#path='/home/zoey/Documents/lectures_and_exercises/period_03/SGN-41006_Signal_Interpretation/Competition_copper/data/'

# get labels of training_set
y=np.loadtxt('train_label.txt')
# obtain the training_set   
X=np.loadtxt('train_set.txt')
X=preprocessing.scale(X)    
#X=float(X)
#y=float(y)

#load the test data
Test_set=np.loadtxt('test_set.txt')
Test_set=preprocessing.scale(Test_set)
name_sort=np.loadtxt('test_set_name.txt',dtype=np.str)
 
# training a model
clf1=svm.SVC(C=7.742, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf1.fit(X,y)
y_pred=clf1.predict(Test_set)
with open('result01.csv','w') as csvfile:
    fieldnames=['Id','Prediction']
    writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()
    j=np.arange(0,len(y_pred),1)
    for i in j:
        writer.writerow({'Id':name_sort[i],'Prediction':int(y_pred[i])})
    
