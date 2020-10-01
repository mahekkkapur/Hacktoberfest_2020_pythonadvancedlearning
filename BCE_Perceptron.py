# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 02:19:17 2020

@author: Hridayesh Singh
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
THRESHOLD = 0.2

class PERCEPTRON :
    def __init__(self, features, target) :
        self.features = features
        self.target = target
        self.bias = 1
        self.weights = np.ones((1,7))
        
    def scale(self, features) :
        return StandardScaler().fit_transform(features)
        
    def preActivation(self, features) :
        self.pAct = (np.dot(self.weights, features) + self.bias)
    
    def activationValue(self) :
        activ = 1/(1+np.exp(-self.pAct))
        return activ
    
    def activation(self, ac) :
        return 1 if ac >= THRESHOLD else 0
    
    def findLoss(self, target, predicted) : 
        # Binary cross entropy
        e1 = np.negative(np.dot(target, np.log2(predicted)))
        e2 = np.negative(np.dot(np.subtract(1, target), np.log2(np.subtract(1, predicted))))
        loss = np.sum((e1, e2))
        return loss
    
    def prediction(self, epochs, lr) :
        out_hist = []
        accuracy = []
        max_accuracy = 0
        epoch_loss = 10
        i = 0
        scaled_features = self.scale(self.features)
        
        while epochs > 0 and epoch_loss >= 1 :
            epoch_loss = 0
            for x,y in zip(scaled_features, self.target) :
                self.preActivation(x)
                actVal = self.activationValue()
                output = self.activation(actVal)
                out_hist.append(output)
                epoch_loss += self.findLoss(y, actVal)
                self.weights[0][0] -= (lr*(output-y)*x[0])
                self.weights[0][1] -= (lr*(output-y)*x[1])
                self.weights[0][2] -= (lr*(output-y)*x[2])
                self.weights[0][3] -= (lr*(output-y)*x[3])
                self.weights[0][4] -= (lr*(output-y)*x[4])
                self.weights[0][5] -= (lr*(output-y)*x[5])
                self.weights[0][6] -= (lr*(output-y)*x[6])
            accuracy.append(accuracy_score(out_hist, self.target))
            if(accuracy[i] > max_accuracy) :
                max_accuracy = accuracy[i]
            out_hist.clear()
            epochs -= 1
            i += 1
        return max_accuracy
    
    
data = pd.read_csv('E:/Machine Learning/GClassroom Content/mobile_cleaned-1549119762886.csv')
data.head()
FEATURES = data[['stand_by_time', 'screen_size', 'battery_capacity', 'processor_rank', 'brand_rank', 'aperture', 'price']]
TARGET_PREDICTION = data[['is_liked']].values
perc = PERCEPTRON(FEATURES, TARGET_PREDICTION)
result = perc.prediction(1000, 0.05)
print(result)    
        