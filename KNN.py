# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:10:19 2019

@author: msouf
"""
import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass
    def train(self,X,Y):
        self.xtr=X
        self.ytr=Y
    
    def predict(self,X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        
        for i in xrange(num_test):
            distances = np.sum(np.abs(self.Xtr-X[i,:]), axis = 1)
            min_index= np.argmin(distances)
            Ypred[i]=self.ytr[min_index]
            
        return Ypred
    