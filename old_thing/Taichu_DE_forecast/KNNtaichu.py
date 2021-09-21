import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding='utf-8-sig',index_col=0)
####Function#############################################################################################################
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
    return Root_error
####Main################################################################################################################
x = data.iloc[:,16]
x_2 = data.iloc[:,17]
temp = data.iloc[:,2]
RH = data.iloc[:,4]
wind_speed=data.iloc[:,5]
sunshine =data.iloc[:,11]
hour=data.loc[:,'hour']
date = data.loc[:,'date']
x = pd.DataFrame({'value':x})
x_2 = pd.DataFrame({'value':x_2})
x_2 = create_lags(x_2,[1,2])
x = create_lags(x,[0,1])

X = pd.concat([x,x_2.iloc[:,1:],temp,RH,wind_speed,sunshine,hour,date],axis=1)
X=X.dropna()
y=X.iloc[:,0]
X = X.iloc[:,1:]
train_idx = int(len(X) * .9)
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]

# neigh = KNeighborsClassifier(n_neighbors=50,weights='distance')
k=np.arange(2,200,1)
error_dec =[]
# for i in k:
#     neigh = KNeighborsRegressor(n_neighbors=i,weights='uniform')
#     model= neigh.fit(X_train,y_train)
#     y_pre =model.predict(X_test)
#     print(i)
#     error=resultcal(y_test, y_pre)
#     error_dec += [error]
neigh = KNeighborsRegressor(n_neighbors=81,weights='uniform')
model= neigh.fit(X_train,y_train)
y_pre =model.predict(X_test)
error=resultcal(y_test, y_pre)
    # error_dec += [error]

# plt.figure()
# plt.plot(k,error_dec)
# plt.show()
#
# # Predicted class
lw = 2 #line width
plt.scatter(y_test.values,y_pre, color='darkorange', label='predict')
plt.plot(y_test.values, y_test.values, color='c', lw=lw)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('KNN')
plt.legend()

result = pd.DataFrame([y_test.values, y_pre], index=['actual', 'predict'])
result = result.T
result.plot()

plt.show()
