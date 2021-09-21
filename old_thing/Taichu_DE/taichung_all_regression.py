import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import neighbors
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun2.csv',encoding='utf-8-sig',index_col=0)
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
x = pd.concat([x,temp,RH,wind_speed,sunshine,hour,date,month],axis=1)
# x = pd.concat([x,x_2.iloc[:,1:],temp,RH,wind_speed,sunshine,hour,date,month,GloblRad],axis=1)
x = x.dropna()

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
# regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4))
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),
                          n_estimators=300)
n_neighbors = 12
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')

#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=3)
# y_rbf10 = svr_rbf10.fit(X_train, y_train).predict(X_test)
y_SVR = svr_rbf1.fit(X_train, y_train).predict(X_test)
model_tree = regr_2.fit(X_train, y_train)
y_tree = model_tree.predict(X_test)
y_KNN = knn.fit(X_train, y_train).predict(X_test)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)

# ###############################################################################
# look at the results
lw = 2 #line width
y_pre1 = y_SVR.reshape(-1,1)
plt.figure()
plt.plot(y_test.values, y_test.values, color='k', lw=lw)
plt.scatter(y_test.values,y_pre1, color='darkorange', label='Support Vector Regression')
plt.scatter(y_test, y_tree, color="g", label="Decision Tree Regression")
plt.scatter(y_test,y_KNN, c='c', label='K-Nearest Neighbors  Regression')


# plt.plot(X_test, y_rbf10, color='navy', lw=lw, label='RBF gamma=10.0')
# plt.plot(X_test, y_rbf1, color='c', lw=lw, label='RBF gamma=1.0')

#plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('measured')
plt.ylabel('predicted')
plt.title('Regression')
plt.legend()

result=pd.DataFrame(y_test.values,columns=['actual power'],index=y_test.index)
result_2 = pd.DataFrame(y_pre1,columns=['SVR'],index=y_test.index)
result_3 = pd.DataFrame(y_tree,columns=['Decision Tree'],index=y_test.index)
result_4 = pd.DataFrame(y_KNN,columns=['KNN'],index=y_test.index)
# result_3 = pd.DataFrame(y_pre10,columns=['predict power_10'])
# result = pd.concat([result,result_2,result_3])
result = pd.concat([result,result_2,result_3,result_4],axis=1)
result.plot()

print(result)
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

SVR_MSE = mean_squared_error(y_test, y_pre1)
SVR_RMSE =SVR_MSE**0.5
SVR_MAE  = mean_absolute_error(y_test, y_pre1)
SVR_MAPE = MAPE(y_test,y_SVR)
SVR_R_2 = r2_score(y_test, y_pre1)
SVR_MRE = MRE(y_test,y_SVR)

Tree_MSE = mean_squared_error(y_test, y_tree)
Tree_RMSE =Tree_MSE**0.5
Tree_MAE = mean_absolute_error(y_test, y_tree)
Tree_MAPE =MAPE(y_test, y_tree)
Tree_R_2 = r2_score(y_test, y_tree)
Tree_MRE = MRE(y_test,y_tree)

KNN_MSE = mean_squared_error(y_test, y_KNN)
KNN_RMSE =KNN_MSE**0.5
KNN_MAE =  mean_absolute_error(y_test, y_KNN)
KNN_MAPE = MAPE(y_test,y_KNN)
KNN_R_2 = r2_score(y_test,y_KNN)
KNN_MRE = MRE(y_test,y_KNN)

MSE = [SVR_MSE,Tree_MSE,KNN_MSE]
RMSE =[SVR_RMSE,Tree_RMSE,KNN_RMSE]
MAE = [SVR_MAE,Tree_MAE,KNN_MAE]
MAPE_s =[SVR_MAPE,Tree_MAPE,KNN_MAPE]
R_2 = [SVR_R_2,Tree_R_2,KNN_R_2]
MRE_s = [SVR_MRE,Tree_MRE,KNN_MRE]
print('Test MSE: ' , MSE)
print('Test RMSE: ' , RMSE)
print('Test MAE:',MAE)
print('Test MAPE:',MAPE_s)
print('Test R^2:',R_2)
print('Test MRE:',MRE_s)

plt.show()


