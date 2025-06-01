import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv("Position_salaries.csv")
x = dataset.iloc[:, 1:-1].values  
y = dataset.iloc[:, -1].values


dt = DecisionTreeRegressor(random_state=0)
dt.fit(x, y)


truth = dt.predict([[6.5]])
print("Prediction for level 6.5: ")



x_grid = np.arange(min(x),max(x),0.1).reshape(-1,1)

plt.scatter(x,y,color ="red")
plt.plot(x_grid,dt.predict(x_grid),color = "blue")
plt.title("Decision tree regression")
plt.xlabel("level")
plt.ylabel("salary")
plt.grid(True)
plt.show()