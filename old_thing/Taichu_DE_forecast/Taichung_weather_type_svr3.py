import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
import matplotlib.pyplot as plt

####Function#############################################################################################################
def create_lags(df, N,s):
    for i in N:
        df[s+'Lag' + str(i+1)] = df.value.shift(i+1)
    return df
####Function#############################################################################################################
def svr_train(X_train,y_train,E,E_1):
    # fit and predict
    svr_rbf1 = SVR(kernel='rbf', C=E, gamma=E_1)
    # Fit regression model
    y_rbf1 = svr_rbf1.fit(X_train, y_train)
    return y_rbf1
####Function#############################################################################################################
def type_w(X,savebase):
    savebase[int(X.weather_type)] += [X.name]
    return savebase

    ####Function#############################################################################################################
def type_index(X,savebase):
    if X.hour < 11:
        savebase[0:3] = type_w(X,savebase[0:3])
    elif X.hour > 14:
        savebase[6:9] = type_w(X,savebase[6:9])
    else:
        savebase[3:6] = type_w(X,savebase[3:6])

    return savebase
####Function#############################################################################################################
def train_Xy(X):
    train_y = X.iloc[:,0]
    train_X = X.iloc[:,1:5]
    return train_X,train_y
####Function#############################################################################################################
def predict_type(X,svr_mode):
    T_value = pd.DataFrame(X[1:5])
    y_pre = svr_mode[int(X.weather_type)].predict(T_value.T)
    return y_pre

####Function#############################################################################################################
def predict_svr(X,svr_mode):
    if X.hour<11:
        y_pre = predict_type(X,svr_mode[0:3])
    elif X.hour>14:
        y_pre =predict_type(X,svr_mode[6:9])
    else:
        y_pre = predict_type(X,svr_mode[3:6])

    return y_pre

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
####Function#############################################################################################################
def draw_result(y_test,y_pred):
    # look at the results
    lw = 2 #line width
    plt.figure()
    plt.scatter(y_test.values,y_pred, color='darkorange', label='predict')
    plt.plot(y_test.values, y_test.values, color='c', lw=lw)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title('Support Vector Regression')
    plt.legend()
    result = pd.DataFrame([y_test.values, y_pred], index=['actual', 'predict'])
    result = result.T
    result.plot()
####Main#############################################################################################################
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_cloudtype.csv',encoding= 'utf-8-sig',index_col=0)
power = data.loc[:,'power(KWH)']
power = pd.DataFrame({'value': power})
power = create_lags(power, [0,1],'power(KWH)')
solar = data.loc[:,'solarirrandance']
solar = pd.DataFrame({'value':solar})
solar = create_lags(solar,[1,2],'solar')
hour = data.hour
weather_type = data.weather_type
data_cl = pd.concat([power,solar.iloc[:,1:],hour,weather_type],axis=1)
data_cl = data_cl.dropna()

train_idx = int(len(data_cl) * 0.9)
# create train and test data
X_train, X_test = data_cl[:train_idx], data_cl[train_idx:]
train_dex=[]
for i in range(0,3):
    train_dex +=[[]]

for i in range(0,len(X_train)):
    train_dex = type_w(X_train.iloc[i,:],train_dex)

Svr_model=[]
for i in range(0,len(train_dex)):
    X = X_train.loc[train_dex[i],:]
    X_svr,y_svr = train_Xy(X)
    svr_T = svr_train(X_svr,y_svr,1000,0.0001)
    Svr_model += [svr_T]

predict_y =np.ndarray([])
for i in range(0,len(X_test)):
    predict_value = predict_type(X_test.iloc[i,:],Svr_model)
    predict_y = np.append(predict_y,predict_value)



predict_y = predict_y[1:]
test_y = X_test.iloc[:,0]
resultcal(test_y,predict_y)
draw_result(test_y,predict_y)
plt.show()

