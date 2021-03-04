import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun.csv',encoding='utf-8-sig',index_col=0)
x = data.loc[:,'power(KWH)']
x_2 = data.loc[:,'solarirrandance']
temp = data.loc[:,'Temperature']
RH = data.loc[:,'RH']
wind_speed=data.loc[:,'WS']
sunshine =data.loc[:,'SunShine']
GloblRad = data.loc[:,'GloblRad']
# cloud_cover = data.iloc[:,15]
hour=data.loc[:,'hour']
date =data.loc[:,'date']
month = data.loc[:,'month']
x = pd.DataFrame({'value':x})
x_2 = pd.DataFrame({'value':x_2})

def create_lags(df, N):
    for i in N:
        df['Lag' + str(i+1)] = df.value.shift(i+1)
    return df
x = create_lags(x,[0,1,2])
x_2 = create_lags(x_2,[0])
# x = pd.concat([x,x_2.iloc[:,1:],temp,RH,UVI],axis=1)
# x = pd.concat([x,x_2.iloc[:,1:],temp,RH,cloud_cover],axis=1)
# x = pd.concat([x,x_2.iloc[:,1:],temp,RH,sunshine],axis=1)
# x = pd.concat([x,x_2.iloc[:,1:],temp,RH],axis=1)
x = pd.concat([x,x_2.iloc[:,1:],temp,RH,wind_speed,sunshine,GloblRad,hour,date,month],axis=1)
x = x.dropna()

y = x.iloc[:, 0]
X = x.iloc[:, 1:]

train_idx = int(len(X) * .8)
# create train and test data
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
# fit and predict
###############################################################################
# Fit regression model
# svr_rbf10 = SVR(kernel='rbf',C=100, gamma=10.0)
svr_rbf1 = SVR(kernel='rbf', C=1000, gamma=0.00001)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=3)
# y_rbf10 = svr_rbf10.fit(X_train, y_train).predict(X_test)
y_rbf1 = svr_rbf1.fit(X_train, y_train).predict(X_test)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)

# ###############################################################################
# look at the results
lw = 2 #line width
y_pre1 = y_rbf1.reshape(-1,1)
plt.scatter(y_test.values,y_pre1, color='darkorange', label='predict')
# plt.plot(X_test, y_rbf10, color='navy', lw=lw, label='RBF gamma=10.0')
# plt.plot(X_test, y_rbf1, color='c', lw=lw, label='RBF gamma=1.0')
plt.plot(y_test.values, y_test.values, color='c', lw=lw)
#plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('measured')
plt.ylabel('predicted')
plt.title('Support Vector Regression')
plt.legend()




result=pd.DataFrame(y_test.values,columns=['actual power'],index=y_test.index)

result_2 = pd.DataFrame(y_pre1,columns=['predict power'],index=y_test.index)
# result_3 = pd.DataFrame(y_pre10,columns=['predict power_10'])
# result = pd.concat([result,result_2,result_3])
result = pd.concat([result,result_2],axis=1)
print(result)
result.plot()
#
#
error = mean_squared_error(y_test, y_pre1)
print('Test MSE: %.3f' % error)
Root_error =error**0.5
print('Test RMSE: %.3f' % Root_error)
R_2 = r2_score(y_test,y_pre1)
print('Test R^2: %.3f' % R_2)
Ab_error = mean_absolute_error(y_test,y_pre1)
print('Test MAE: %.3f' % Ab_error)
#
plt.show()
# c = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]
# all_error =pd.DataFrame()
# for i in c :
#     svr_rbf1 = SVR(kernel='rbf', C=100, gamma=i)
#     y_rbf1 = svr_rbf1.fit(X_train, y_train).predict(X_test)
#     y_pre1 = y_rbf1.reshape(-1, 1)
#     MSE = mean_squared_error(y_test, y_pre1)
#     RMSE = MSE ** 0.5
#     MAE = mean_absolute_error(y_test, y_pre1)
#     R_2 = r2_score(y_test, y_pre1)
#     error =pd.DataFrame([R_2,RMSE,MAE],index=['R^2','RMSE','MAE'],columns={i})
#     error = error.T
#     all_error = all_error.append(error)
#     print(i)


# print(all_error)





