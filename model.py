# -*- coding: utf-8 -*-
import numpy as np


class model: 
    #The linear model is represented by the vector A.
    # where the A[0] = b and X[:,0] = 1
	
    #construct and randomly initialize the linear model, with n params.
    def __init__(self, n):
        rand = np.random.rand(n+1)
        self.A = np.array(rand)
        
    #return the linear model    
    def get_param(self):
        return self.A

    #predict the input using the model.
    def predict(self, input):
        return np.matmul(input, self.A)
    
    #calculate the L^2 loss on an input matrix X with targets y.
    def loss(self, X, y):
        predictions = self.predict(X)
        loss = np.square((predictions - y))
        return loss.sum()
    
    #fit the model to the learning data using GD.
    def fit_GD(self, X, y, eta = 0.00001):
        for i in range(10000):
            #get the predictions as batch-vector
            y_pred = self.predict(X)
            batch_difference = np.matmul(X.T , (y - y_pred))
            gradient = (-2/ len(y)) * np.sum(batch_difference, axis = 0 )
            #update parameters
            self.A -= eta * gradient

    #fit the model to the learning data using GD.
    def fit_GD2(self, X, y, eta = 0.00001, eps = 0.1): 
        gradient = 10
        while gradient > eps:
            #get the predictions as batch-vector
            y_pred = self.predict(X)
            batch_difference = np.matmul(X.T , (y - y_pred))
            gradient = (-2/ len(y)) * np.sum(batch_difference, axis = 0 )
            #update parameters
            self.A -= eta * gradient

    #fit model using the analytical solution of the linear model, given as 
    # A = (X.T * X)^-1 * X.T * y
    # The input X is given as a mxn matrix, where m is the # of samples.
    # y is the value vector for the inputs X.
    def fit_analytic(self, X, y):
        self.A = np.matmul(np.linalg.inv( np.matmul(X.T, X) ) , np.matmul(X.T, y))
    

            
        
        

