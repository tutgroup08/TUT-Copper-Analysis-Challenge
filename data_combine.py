# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:03:50 2016

@author: Lingyu
"""


#import csv
#import cv2
import numpy as np

# get labels of training_set
y1=np.loadtxt('train_label.txt')
# obtain the training_set   
X1=np.loadtxt('train_set.txt')
y_add=np.loadtxt('train_label_weMade.txt')
X_add=np.loadtxt('train_set_weMade.txt')
index_weMade=np.loadtxt('train_set_weMade_index.txt')
#print y_add.shape
#print X_add[0].shape
X=[]
y=[]
for i in X1:
    X.append(i)
for i in y1:
    y.append(i)
for i in index_weMade:
    X.append(X_add[i])
    y.append(y_add[i])

X=np.array(X)
y=np.array(y)

np.savetxt("train_set_ori_3_6_8.txt",X)
np.savetxt("train_label_ori_3_6_8.txt",y,fmt='%u')
