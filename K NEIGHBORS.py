# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:26:05 2020

@author: Usuario
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split



df = pd.read_csv('D:/Anaconda/datasets/cancer/breast-cancer-wisconsin.data.txt',header=None)

#CAMBIO DE NOMBRE A LAS COLUMNAS

df.columns= ['name','V1','V2','V3','V4','V5','V6','V7','V8','V9','class']

print(df.head())

df = df.drop('name',1)

df.replace('?',-99999,inplace=True)

y=df['class']
x=df[['V1','V2','V3','V4','V5','V6','V7','V8','V9']]


#VALIDACION CRUZADA MODELO

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()

clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)

sample = np.array([4,2,1,1,1,2,3,2,1]).reshape(1,-1)

print(clf.predict(sample))

