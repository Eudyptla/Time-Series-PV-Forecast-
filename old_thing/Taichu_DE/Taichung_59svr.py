import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

data_0 = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun59.csv',encoding='utf-8-sig',index_col=0)
data_1 = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sunelse.csv',encoding='utf-8-sig',index_col=0)

def create_lags(df, N,s):
    for i in N:
        df[s+'Lag' + str(i+1)] = df.value.shift(i+1)
    return df

def condata(data):
    x = data.iloc[:, 16]
    x_2 = data.iloc[:, 17]
    temp = data.iloc[:, 2]
    RH = data.iloc[:, 4]
    # UVI = data.iloc[:, 14]
    # sunshine = data.iloc[:, 11]
    # cloud_cover = data.iloc[:, 15]
    wind_speed = data.iloc[:, 5]
    hour = data.loc[:, "hour"]
    x = pd.DataFrame({'value': x})
    x_2 = pd.DataFrame({'value': x_2})
    temp = pd.DataFrame({'temp':temp})
    RH =pd.DataFrame({"RH":RH})
    wind_speed =pd.DataFrame({"WD":wind_speed})
    hour = pd.DataFrame({"Hour":hour})
    x = create_lags(x, [0, 1],'power(KWH)')
    x_2 = create_lags(x_2, [0],'solarirrandance')

    x = pd.concat([x, x_2.iloc[:, 1:], temp, RH, wind_speed, hour], axis=1)
    x = x.dropna()
    return x


data_0 = condata( data_0 )
data_1 = condata( data_1 )
# x = pd.concat([x,x_2.iloc[:,1:],temp,RH,UVI],axis=1)
# x = pd.concat([x,x_2.iloc[:,1:],temp,RH,cloud_cover],axis=1)
# x = pd.concat([x,x_2.iloc[:,1:],temp,RH,sunshine],axis=1)
# x = pd.concat([x,x_2.iloc[:,1:],temp,RH],axis=1)
# x = pd.concat([x,x_2.iloc[:,1:],temp,RH,wind_speed],axis=1)
# x = pd.concat([x,temp,RH,wind_speed],axis=1)
# x = pd.concat([x,temp,RH,wind_speed,hour],axis=1)
def split_data(x):
    y = x.iloc[:, 0]
    X = x.iloc[:, 1:]
    train_idx = int(len(X) * .83)
    # create train and test data
    X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
    return X_train, y_train, X_test, y_test
data_0_train,target_0_train,data_0_test,target_0_test = split_data(data_0)
data_1_train,target_1_train,data_1_test,target_1_test = split_data(data_1)

data_train = pd.concat([data_0_train,data_1_train],axis=0,sort=False)
target_train = pd.concat([target_0_train,target_1_train],axis=0,sort=False)
data_test = pd.concat([data_0_test,data_1_test],axis=0,sort=False)
target_test = pd.concat([target_0_test,target_1_test],axis=0,sort=False)


def svr_train(X_train,y_train,X_test):
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
    return y_rbf1

    # ###############################################################################
data_0_pre = svr_train(data_0_train,target_0_train,data_0_test)
data_1_pre = svr_train(data_1_train,target_1_train,data_1_test)
data_s_pre = np.concatenate((data_0_pre,data_1_pre),axis=0)
data_pre = svr_train(data_train,target_train,data_test)
def draw_result(y_test,y_pred):
    # look at the results
    lw = 2 #line width
    y_pre1 = y_pred.reshape(-1,1)
    plt.figure()
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
    result.plot()

draw_result(target_test,data_s_pre)
draw_result(target_test,data_pre)

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


resultcal(target_test,data_s_pre)
resultcal(target_test,data_pre)
plt.show()



