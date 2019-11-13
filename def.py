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
        return sigp(self.v)*dx

        