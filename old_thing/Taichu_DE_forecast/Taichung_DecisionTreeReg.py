import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
import numpy as np
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding='utf-8-sig',index_col=0)

def create_lags(df, N):
    for i in N:
        df['Lag' + str(i+1)] = df.value.shift(i+1)
    return df
####Function#############################################################################################################
def MAPE(x,y):
    n = len(x)
    x = x.values
    d = (sum(abs((x-y)/x)))/n
    return d
####Function#############################################################################################################
def MRE(x,y):
    n=len(x)
    x=x.values
    Xmax = max(x)
    d = sum(abs((x-y)/Xmax))/n
    return d
####Function#############################################################################################################
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
x = data.loc[:,'power(KWH)']
x_2 = data.loc[:,'solarirrandance']
temp = data.loc[:,'Temperaure']
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
x_2 = create_lags(x_2,[1,2])
x = create_lags(x,[0,1])
# cloud_cover = pd.DataFrame({'Cloud Amoun':cloud_cover})

# X = pd.concat([x,x_2.iloc[:,1:],temp,RH,wind_speed,hour],axis=1)
X = pd.concat([x,x_2.iloc[:,1:],temp,RH,wind_speed,hour],axis=1)
X=X.dropna()
y=X.iloc[:,0]
X = X.iloc[:,1:]
train_idx = int(len(X) * .9)
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=9)

# regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
#                           n_estimators=100)

# model_1 = regr_1.fit(X_train, y_train)
model_2 = regr_1.fit(X_train, y_train)

# Predict
# y_1 = model_1.predict(X_test)
y_2 = model_2.predict(X_test)

# Plot the results



def draw_result(y_test,y_pred):
    # look at the results
    lw = 2 #line width
    plt.figure()
    plt.scatter(y_test.values,y_pred, color='darkorange', label='predict')
    plt.plot(y_test.values, y_test.values, color='c', lw=lw)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title('Decision Tree Regression')
    plt.legend()
    result = pd.DataFrame([y_test.values, y_pred], index=['actual', 'predict'])
    result = result.T
    result.plot()

resultcal(y_test,y_2)
draw_result(y_test,y_2)
plt.show()