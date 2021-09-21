import numpy as np
from numpy import *
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt




x = np.linspace(-5,5,200)
siny = np.sin(x)
X = mat(x).T
y = siny + np.random.rand(1,len(siny))*1.5
clf_1 = DecisionTreeRegressor(max_depth=4).fit(X,y[0])
clf_2 = DecisionTreeRegressor(max_depth=6).fit(X,y[0])
clf_3 = DecisionTreeRegressor(max_depth=8).fit(X,y[0])
X_test = np.arange(-5.0,5.0,0.05)[:,np.newaxis]
yp_1 = clf_1.predict(X_test)
yp_2 = clf_2.predict(X_test)
yp_3 = clf_3.predict(X_test)


plt.figure()
plt.scatter(x,y,c="orange",label="data")
plt.plot(X_test,yp_1,c='r',label="max_depth=4",linewidth=2)
plt.plot(X_test,yp_3,c='blue',label="max_depth=8",linewidth=2,linestyle=':')
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend(loc='upper right')
plt.show()
