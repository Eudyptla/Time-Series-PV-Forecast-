import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt

def create_lags(df, N):
    for i in N:
        df['Lag' + str(i+1)] = df.value.shift(i+1)
    return df

def MRE(x,y):
    n=len(x)
    Xmax = max(x)
    d = sum(abs((x-y)/Xmax))/n
    return d

def MAPE(x,y):
    n = len(x)
    d = (sum(abs((x-y)/x)))/n
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



data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding='utf-8-sig',index_col=0)
power_data = data.iloc[:,16]
solar = data.iloc[:,17]
power = pd.DataFrame({'value': power_data})
power = create_lags(power,[0])
solar = pd.DataFrame({'value':solar})
solar = create_lags(solar,[1])
power = pd.concat([power,solar.iloc[:,-1]],axis=1)
power=power.dropna()
print(power)
X = power.iloc[:,1:].values
y = power.iloc[:,0].values
train_idx = int(len(X) * .9)
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]

olsmod = sm.OLS(y_train, X_train)
olsres = olsmod.fit()
print(olsres.summary())

ypred = olsres.predict(X_test)
lw = 2 #line width
plt.scatter(y_test,ypred, color='darkorange', label='predict')
plt.plot(y_test, y_test, color='c', lw=5)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('Ordinary Least Squares')
plt.legend()

result =pd.DataFrame([y_test,ypred],index=['actual','predict'])
result=result.T
result.plot()


resultcal(y_test,ypred)

plt.show()