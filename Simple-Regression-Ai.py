import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv("Salary_data.csv")
x = dataset.iloc[:,[0]].values
y = dataset.iloc[:,1].values


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

plt.scatter(x_train,y_train,color = "red")
plt.plot(x_train,lr.predict(x_train),color = "blue")
plt.title("salary vs years of expireince Train")
plt.xlabel("years")
plt.ylabel("salary")
plt.show()

plt.scatter(x_test,y_test,color = "red")
plt.plot(x_test,y_pred,color = "blue")
plt.title("salary vs years of expireince Test")
plt.xlabel("years")
plt.ylabel("salary")
plt.show()
