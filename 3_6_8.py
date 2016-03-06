# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:36:36 2016

@author: Lingyu
"""
import numpy as np
import csv
import glob
import cv2
import os
from skimage import feature
#from sklearn import preprocessing
# obtain the training_set   
X=[]
y1=[]
y=[]
index_weMade=[]
#name_value=[]


# get labels of training_set
#y1=np.loadtxt('train_label.txt')

# get train_set
file_lists=glob.glob('data/train_weMade/*')
i=0
for image in file_lists:
    #print i,image
    c=os.path.basename(image).split(".")[0]
    index_weMade.append(i)
    i=i+1
    
    d=int(c.split("_")[0])
    y.append(d)
    img=cv2.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lbp_image=feature.local_binary_pattern(gray,8,3)
    b=lbp_image.max()+1
    #histogram=[np.histogram(iim[...,channel])[0] for channel in [0,1,2]]
    histogram=np.histogram(lbp_image,normed=True,bins=b,range=(0,b))[0] 
    #print len(histogram)
    #break
    X.append(histogram) 
#print index_weMade   
np.savetxt("train_set_weMade.txt",X)
np.savetxt("train_label_weMade.txt",y,fmt='%u')
np.savetxt("train_set_weMade_index.txt",index_weMade,fmt='%u') 
        

    
