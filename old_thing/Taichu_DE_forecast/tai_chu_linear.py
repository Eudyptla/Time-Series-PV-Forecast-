import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd

data_0 = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_power_weather.csv',encoding='utf-8-sig',index_col=0)
x = data_0.iloc[1:,17].fillna(0).values #late 1 hour
y = data_0.iloc[:-1,17].fillna(0).values #late 1 hour
X = x.reshape(-1,1)
y = y.reshape(-1,1)
train_idx = int(len(X) * .8)
# create train and test data
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
# fit and predict
###############################################################################
# Fit regression model
# svr_rbf10 = SVR(kernel='rbf',C=100, gamma=10.0)
svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=3)
# y_rbf10 = svr_rbf10.fit(X_train, y_train).predict(X_test)
# y_rbf1 = svr_rbf1.fit(X_train, y_train).predict(X_test)
y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)

###############################################################################
# look at the results
lw = 2 #line width
plt.scatter(X_test,y_test, color='darkorange', label='data')
plt.hold('on')
# plt.plot(X_test, y_rbf10, color='navy', lw=lw, label='RBF gamma=10.0')
# plt.plot(X_test, y_rbf1, color='c', lw=lw, label='RBF gamma=1.0')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

# y_pre10 = y_rbf10.reshape(-1,1)
y_pre1 = y_lin.reshape(-1,1)
result=pd.DataFrame(y_test[1:],columns=['actual power'])
result_2 = pd.DataFrame(y_pre1[:-1],columns=['predict power'])
# result_3 = pd.DataFrame(y_pre10,columns=['predict power_10'])
# result = pd.concat([result,result_2,result_3])
result = pd.concat([result,result_2],axis=1)
result.plot()
plt.show()
print(result)


