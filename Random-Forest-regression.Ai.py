import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


regressor = RandomForestRegressor(n_estimators=1000000000,random_state=0)
regressor.fit(x,y)



x_grid = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_pred = regressor.predict(x_grid)

plt.scatter(x,y,color = "red")
plt.plot(x_grid,y_pred,color = "blue")
plt.title("RandomForest regression")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()


z = regressor.predict([[6.5]])
print(z)