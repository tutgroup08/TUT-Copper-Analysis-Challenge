# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 18:15:50 2016

@author: Lingyu
"""


import numpy as np
import csv
import glob
import cv2
import os
from skimage import feature

# obtain the training_set   
X=[]
y1=[]
y=[]
name=[]

with open('train.csv') as csvfile:
    reader=csv.DictReader(csvfile)
    for row in reader:
        y1.append(int(row['Prediction']))


# get train_set
file_lists=glob.glob('data/train/*')

for image in file_lists:
    c=os.path.basename(image).split(".")[0]
    name.append(c)
    d=int(c.split("_")[1])
    y.append(y1[d])
    img=cv2.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lbp_image=feature.local_binary_pattern(gray,8,3)
    b=lbp_image.max()+1
    #histogram=[np.histogram(iim[...,channel])[0] for channel in [0,1,2]]
    histogram=np.histogram(lbp_image,normed=True,bins=b,range=(0,b))[0] 
    X.append(histogram)
    
#load the test data  
name1=[]
name_value=[]
y_pred=[]
Test_set=[]

file_lists_test=glob.glob('data/test/*') 
for image in file_lists_test:
    c=os.path.basename(image)
    name1.append(c)
    c=os.path.basename(image).split('.')[0]
    d=int(c.split("_")[1])
    name_value.append(d)
    img=cv2.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lbp_image=feature.local_binary_pattern(gray,8,3)
    b=lbp_image.max()+1
    histogram=np.histogram(lbp_image,normed=True,bins=b,range=(0,b))[0] 
    Test_set.append(histogram)

indexes=[]
Test_sort=[]
name_sort=[]
indexes=sorted(name_value,key=lambda k: name_value[k])
for n in indexes:
    Test_sort.append(Test_set[n])
    name_sort.append(name1[n])

np.savetxt("test_set.txt",Test_sort)   
np.savetxt("test_set_name.txt",name_sort,fmt='%s')  
np.savetxt("train_set.txt",X)
np.savetxt("train_label.txt",y,fmt='%u')
        

    
