# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:41:25 2019

@author: msouf
"""

import functions as f
import numpy as np
import matplotlib.pyplot as plt

#DATAS
labda=0.1
X=np.random.randn(50,2) # x1 and x2 #Add column of 1 for the bias
X=np.concatenate(X,np.asarray([1]*len(X)))
Y=2*np.random.randint(0,2,size=(50))-1
W=np.random.randn(1,2)*0.01    
bias=0.1

#Plot datas
f.showDatas(X,W,Y,bias)

e=0

p=f.perceptron(X,W)

loss=p.forward()

#for i in range(len(X)):
#    #forward pass
#    v=np.dot(W,X[i,:])+bias
#    y=sig(v)
#    
#    #loss function
#    l=loss(y,Y[i])
#    plt.scatter(e,l,c='green')
#    e+=1
#    #backward pass
#    dl=sigp(v)*(1/2)*(y-Y[i])
#    
#    dW= np.asarray([j*labda*dl for j in X[i]])
#    #W=np.transpose(W)
#    W = np.subtract(W,dW)
#    bias-=labda*dl
#
#plt.show()
#
#showDatas(X,W,Y,bias)