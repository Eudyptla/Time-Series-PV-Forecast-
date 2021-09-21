import numpy as np
from numpy import *
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt




x = np.linspace(-5,5,200)
siny = np.sin(x)
X = mat(x).T
y = siny + np.random.rand(1,len(siny))*1.5
clf_1 = KNeighborsRegressor(n_neighbors=3,weights='uniform').fit(X,y[0])
clf_2 = KNeighborsRegressor(n_neighbors=30,weights='uniform').fit(X,y[0])
X_test = np.arange(-5.0,5.0,0.05)[:,np.newaxis]
yp_1 = clf_1.predict(X_test)
yp_2 = clf_2.predict(X_test)



plt.figure()
plt.scatter(x,y,c="orange",label="data")
plt.plot(X_test,yp_2,c='r',label="K=30",linewidth=2)
plt.plot(X_test,yp_1,c='blue',label="K=3",linewidth=2,linestyle=':')
plt.xlabel("data")
plt.ylabel("target")
plt.title("KNN Regression")
plt.legend(loc='upper right')
plt.show()
