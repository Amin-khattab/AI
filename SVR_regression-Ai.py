import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv("Position_Salaries.csv")
x =dataset.iloc[:,1:-1].values
y =dataset.iloc[:,-1].values

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(len(y),1))

regressor = SVR(kernel="rbf")
regressor.fit(x,y)

scaled = regressor.predict(sc_x.transform([[6.5]]))
original = sc_y.inverse_transform(scaled.reshape(-1,1))

X = sc_x.inverse_transform(x)
Y = sc_y.inverse_transform(y)

z = sc_y.inverse_transform(regressor.predict(x).reshape(-1,1))

plt.scatter(X,Y,color ="red")
plt.plot(X,z,color = "blue")
plt.title("SVR")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()