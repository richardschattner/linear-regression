# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from model import model
from sklearn.model_selection import train_test_split

os.chdir(r'C:\things\test\test_project')
X = pd.read_csv('Boston.csv')

#normalize the training data
mean = X.mean()
std = X.std()

def normalize(X):
    return ((X-mean)/std)

def unnormalize(X):
    return (X * std) + mean
    

# the target variable 'y' shall be the last attribute 'MEDV'.
#This is the median value of owner-occupied homes in Boston suburbs.  
#The input matrix 'X' is a m x (n+1) matrix. Here m = 506 samples and n = 13

X = normalize(X)
y = X['MEDV'].to_numpy()

ones = pd.DataFrame(np.ones(len(y)))
X = X.iloc[: , :-1]
X = ones.join(X).to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X ,y,  test_size= 0.25)


#initialize the model
model = model(13)

#fit with gradient descent and calculate the training loss
model.fit_GD(X_train, y_train)
print(model.loss(X_test, y_test))

model.fit_GD2(X_train, y_train)
print(model.loss(X_test, y_test))


model.fit_analytic(X_train, y_train)
print(model.loss(X_test, y_test))



