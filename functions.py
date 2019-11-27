# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:50:30 2019

@author: msouf
"""
import numpy as np
import math as m
import matplotlib.pyplot as plt

loss=lambda y,yd:m.pow(y-yd,2)
sig=lambda v:(m.exp(v)-m.exp(-v))/(m.exp(v)+m.exp(-v))
sigp=lambda v:sig(v)*(1-sig(v))

def mlp1Layer(X,W):
    #forwardPass
    bias=W[-1]    
    #forward pass
    v=np.dot(W,X)+bias
    y=sig(v)
    
    return y

def showDatas(X,W,Y,bias):
    #Plot datas
    fig=plt.figure()
    ax=plt.axes()
    x= np.linspace(0,10,1000)
    ax.plot(x,(-W[0,0]*x-bias)/W[0,1])
    
    for i in range(len(X)):  
        if Y[i]==1:
            ax.scatter(X[i,0],X[i,1],c='red')
        else:
            ax.scatter(X[i,0],X[i,1],c='blue')
            
    
    plt.show()

class perceptron:
    
    #weight=0;
    
    def __init__(self,x,w):
        self.weight=w
        self.features=x
        
        
    def forward(self):
        self.v=np.dot(self.weight,self.features)
        return sig(self.v)
    
    def backwardgradient(self,dx):
        #dx=dv*dy
        self.dv=sigp(self.v)*dx
        return self.dv
    
    def updateWeight(self):
        self.weight=self.weight+self.dv

        