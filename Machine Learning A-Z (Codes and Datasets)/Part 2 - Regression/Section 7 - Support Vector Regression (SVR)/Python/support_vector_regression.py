# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
X = sc_x.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(sc_x.transform([[6.5]]))  # so that the value will be on the same scale as before

print(sc_y.inverse_transform([y_pred]))  # reversing the scaling transformation to give the real number
reg = regressor.predict(X).reshape(len(X), 1)


# Visualising the SVR results
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(reg), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_x.inverse_transform(X)), max(sc_x.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
reg1 = regressor.predict(sc_x.transform(X_grid)).reshape(len(X_grid), 1)
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_x.inverse_transform(reg1), color='blue')
plt.title('Truth or Bluff (Fine Poly)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()
