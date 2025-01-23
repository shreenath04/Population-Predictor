import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('NCHS_-_Births_and_General_Fertility_Rates__United_States.csv')
print(data.head())
print()
print(data.describe())
print()
print(data.info())
print()
print()
print()

data['Year'] = data['Year']/100
print(data['Year'])

x = data[["Year","General Fertility Rate"]]
y = data["Crude Birth Rate"]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

x_train = np.array(X_train)
y_train = np.array(Y_train)
x_val = np.array(X_test)
y_val = np.array(Y_test)

from sklearn import svm

model_regression = svm.LinearSVR()
model_regression.fit(x_train,y_train)

error = 0
n = len(x_val)
y_pred = model_regression.predict(x_val)
y_pred = np.array(y_pred)

error = np.array((y_pred-y_val)**2)
MSE = error/n
RMSE = MSE**(1/2)
MAE = error = np.array(y_pred-y_val)
print(MSE)
print()
print()
print(model_regression.predict([[2019,58.2]]))
plt.plot(x_val[:,0],y_pred,color='red')
plt.scatter(x_train[:,0],y_train,color='blue')
plt.show()

plt.plot(x_val[:,1],y_pred,color='red')
plt.scatter(x_train[:,1],y_train,color='blue')
plt.show()