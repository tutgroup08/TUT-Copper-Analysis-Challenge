# TUT-Copper-Analysis-Challenge
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:09:20 2016
@author: Lingyu
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import os.path
#from imutils import paths
from skimage import feature
# import skimage.feature.local_binary_pattern as LBP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV
#from sklearn.neural_network import 

#path='/home/zoey/Documents/lectures_and_exercises/period_03/SGN-41006_Signal_Interpretation/Competition_copper/data/'

X=[]
y1=[]
y=[]

# get labels of training_set
with open('train.csv') as csvfile:
    reader=csv.DictReader(csvfile)
    for row in reader:
        y1.append(int(row['Prediction']))

# obtain the training_set       
file_lists=glob.glob('data/train/*')

for image in file_lists:
    #print image,'\n'
    c=os.path.basename(image).split(".")[0]
    d=int(c.split("_")[1])
    #c=re.split("_ .",os.path.basename(image))[0]
    y.append(y1[d])
    im=plt.imread(image)
    #iim=im.reshape(1,512*512*3)
    im_mean=np.mean([im[:,:,0],im[:,:,1],im[:,:,2]],axis=0)
   
    #lbp_image=[feature.local_binary_pattern(im[:,:,0],8,5,method='uniform')]
    lbp_image=[feature.local_binary_pattern(im_mean,8,5,method='uniform')]
  
    #histogram=[np.histogram(iim[...,channel])[0] for channel in [0,1,2]]
    histogram=np.histogram(lbp_image,bins=540)[0] 
    #print len(histogram)
    #break
    X.append(histogram)
X=preprocessing.scale(X)    
# training a model
classifiers=[
           (svm.SVC(C=10**12, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False),"Support Vector Machine")
             ]

#Split your data into two parts—80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
i=0
for clf,name in classifiers:
    
    clf.fit(X_train,y_train)
    print accuracy_score(y_test,clf.predict(X_test))


#load the test data  
name=[]
name_value=[]
y_pred=[]
Test_set=[]

file_lists_test=glob.glob('data/test/*') 
for image in file_lists_test:
    #print image,'\n'
    c=os.path.basename(image)
    name.append(c)
    c=os.path.basename(image).split('.')[0]
    d=int(c.split("_")[1])
    name_value.append(d)
    im=plt.imread(image)
    #iim=im.reshape(1,512*512*3)
    im_mean=np.mean([im[:,:,0],im[:,:,1],im[:,:,2]],axis=0)
   
    #lbp_image=[feature.local_binary_pattern(im[:,:,0],8,5,method='uniform')]
    lbp_image=[feature.local_binary_pattern(im_mean,8,5,method='uniform')]
  
    #histogram=[np.histogram(iim[...,channel])[0] for channel in [0,1,2]]
    histogram=np.histogram(lbp_image,bins=540)[0] 
    #print len(histogram)
    #break
    Test_set.append(histogram)
Test_set=preprocessing.scale(Test_set)

indexes=[]
Test_sort=[]
name_sort=[]
indexes=sorted(name_value,key=lambda k: name_value[k])
for n in indexes:
    Test_sort.append(Test_set[n])
    name_sort.append(name[n])

y_pred=clf.predict(Test_sort)
with open('result05.csv','w') as csvfile:
    fieldnames=['Id','Prediction']
    writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
    
    writer.writeheader()
    j=np.arange(0,len(y_pred),1)
    for i in j:
        writer.writerow({'Id':name_sort[i],'Prediction':y_pred[i]})
