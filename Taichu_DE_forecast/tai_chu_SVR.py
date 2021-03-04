import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding='utf-8-sig',index_col=0)

weather_data = data.iloc[:,0:13]
time_data = data.iloc[:,-3:]
power_data = data.iloc[:,16]
solar_Irandance= data.iloc[:,17]
def create_lags(df, N):
    for i in N:
        df['Lag' + str(i+1)] = df.value.shift(i+1)
    return df

power = pd.DataFrame({'value': power_data})
power = create_lags(power,[0,1])
solar_Irandance = pd.DataFrame({'value': solar_Irandance})
solar_Irandance = create_lags(solar_Irandance,[1,2])
data_all = pd.concat([power,weather_data,solar_Irandance.iloc[:,1],time_data],axis=1)
# data_all = pd.concat([power,weather_data,time_data],axis=1)
data_all=data_all.dropna()

X = data_all.iloc[:,1:]
y = data_all.iloc[:,0]
train_idx = int(len(X) * .9)

# create train and test data
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
# fit and predict
###############################################################################
# Fit regression model
# svr_rbf10 = SVR(kernel='rbf',C=100, gamma=10.0)
svr_rbf1 = SVR(kernel='rbf', C=3000, gamma=0.00005)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=3)
# y_rbf10 = svr_rbf10.fit(X_train, y_train).predict(X_test)
y_rbf1 = svr_rbf1.fit(X_train, y_train).predict(X_test)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)

# ###############################################################################
# look at the results
lw = 2 #line width

plt.scatter(y_test.values,y_rbf1, color='darkorange', label='predict')

# plt.plot(X_test, y_rbf10, color='navy', lw=lw, label='RBF gamma=10.0')
# plt.plot(X_test, y_rbf1, color='c', lw=lw, label='RBF gamma=1.0')
plt.plot(y_test.values, y_test.values, color='c', lw=lw)
#plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('measured')
plt.ylabel('predicted')
plt.title('Support Vector Regression')
plt.legend()


def MAPE(x,y):
    n = len(x)
    x = x.values
    d = (sum(abs((x-y)/x)))/n
    return d

def MRE(x,y):
    n=len(x)
    x=x.values
    Xmax = max(x)
    d = sum(abs((x-y)/Xmax))/n
    return d
def resultcal(y_test,y_pre):
    error = mean_squared_error(y_test, y_pre)
    Root_error =error**0.5
    SVR_MAE  = mean_absolute_error(y_test, y_pre)
    SVR_MAPE = MAPE(y_test,y_pre)
    SVR_R_2 = r2_score(y_test, y_pre)
    SVR_MRE = MRE(y_test,y_pre)
    print('Test MSE: %.3f' % error)
    print('Test RMSE: %.3f' % Root_error)
    print('Test MAE: %.3f' % SVR_MAE)
    print('Test MAPE: %.3f' % SVR_MAPE)
    print('Test MRE: %.3f' % SVR_MRE)
    print('Test R^2: %.3f' % SVR_R_2)

result = pd.DataFrame([y_test.values, y_rbf1], index=['actual', 'predict'])
result = result.T
result.plot()

resultcal(y_test,y_rbf1)
plt.show()




