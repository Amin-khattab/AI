import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("50_Startups.csv")
x =dataset.iloc[:,:-1].values
y =dataset.iloc[:,-1].values

ct = ColumnTransformer(transformers=[("amin",OneHotEncoder(),[3])],remainder="passthrough")
x = ct.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
np.set_printoptions(precision=0)
print(np.concatenate((y_pred.reshape(-1,1), y_test.reshape(-1,1)), axis=1))
