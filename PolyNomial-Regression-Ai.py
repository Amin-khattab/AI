import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Position_Salaries.csv")
x =dataset.iloc[:,1:-1]
y =dataset.iloc[:,-1]

lin_reg = LinearRegression()
lin_reg.fit(x,y)
lin_predict = lin_reg.predict(x)

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)
lin_predict_2 =lin_reg_2.predict(x_poly)

plt.scatter(x,y,color = "red")
plt.plot(x,lin_predict,color = "orange")
plt.title("simple linear reggresion")
plt.xlabel("levels")
plt.ylabel("Salary")
plt.show()

plt.scatter(x,y,color = "red")
plt.plot(x,lin_predict_2,color = "orange")
plt.title("poly linear regression")
plt.xlabel("levels")
plt.ylabel("Salary")
plt.show()

z = lin_reg.predict([[6.5]])
print("prediction for simple is",z)

j = lin_reg_2.predict(poly_reg.transform([[6.5]]))
print("prediction for ploy is ",j)



