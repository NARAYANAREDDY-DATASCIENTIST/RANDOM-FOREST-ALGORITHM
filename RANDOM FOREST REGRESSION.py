# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:37:52 2020

@author: NARAYANA REDDY DATA SCIENTIST
"""
# RANDOM FOREST REGRESSION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the data set
dataset=pd.read_csv('position_salaries.csv')
# divide the data set into independent and depedent variable
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# splitting the data set into train and test data
 """from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.3,ramdom_size=0)"""

# Feature scaling
""" from sklearn.preprocessing import StandardScaler
Sc_x=StandardScaler()
x_train=Sc_x.fit_transform(x_train)
x_test=Sc_x.fit_transform(x_test) """

# Fitting the Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
Regressor=RandomForestRegressor(n_estimators=10,random_state=0)
Regressor.fit(x,y)

# predict the regressor

y_predict=Regressor.predict(x)


# basic graph
plt.scatter(x,y,color='red')
plt.plot(x,Regressor.predict(x), color='blue')
plt.show
# Visualising the RandomForest Regressor Results (Higher Resolution)

x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color='red')
plt.plot(x_grid,Regressor.predict(x_grid), color='blue')
plt.title('TRUTH OR BLUFF (RANDOM FOREST REGRESSOR)')
plt.xlabel('position level')
plt.ylabel('salaries')
plt.show
