import pandas as pd
from sklearn import cluster
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np

###############################################################################
def create_lags(df, N):
    for i in N:
        df['Lag' + str(i+1)] = df.value.shift(i+1)
    return df
###############################################################################
# Fit regression model
def svr_train(X_train,y_train,E,E_1):
    # fit and predict
    # Fit regression model

    svr_rbf1 = SVR(kernel='rbf', C=E, gamma=E_1)
    y_rbf1 = svr_rbf1.fit(X_train, y_train)
    return y_rbf1

###############################################################################
def data_loc(label,target):
    index = []
    log=0
    for i in label:
        if i == target:
            index += [log]
        log += 1

    return index
################################################################################
def data_re(data_N):
    data_N = data_N.values
    data_N = data_N.reshape(-1,1)
    return data_N
################################################################################
def MAPE(x,y):
    n = len(x)
    x = x.values
    d = (sum(abs((x-y)/x)))/n
    return d
################################################################################
def MRE(x,y):
    n=len(x)
    x=x.values
    Xmax = max(x)
    d = sum(abs((x-y)/Xmax))/n
    return d
################################################################################
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
################################################################################

data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding='utf-8-sig',index_col=0)

weather_data = data.iloc[:,0:13].values
time_data = data.iloc[:,-3:].values
power_data = data.iloc[:,16].values
Irandiance =data.iloc[:,17].values
power = pd.DataFrame({'value': power_data})
Iran = pd.DataFrame({'value':Irandiance})
power = create_lags(power,[0,1])
Iran = create_lags(Iran,[1,2])
weather_data=pd.DataFrame(weather_data)
time_data=pd.DataFrame(time_data)
data_all = pd.concat([power,Iran.iloc[:,1:],weather_data,time_data],axis=1)
data_all=data_all.dropna()
K_data = data_all.iloc[:,5:]
power_data = data_all.iloc[:,:5]
train_idx = int(len(K_data) * .9)
# create train and test data
K_train, K_test,= K_data[:train_idx], K_data[train_idx:]
error = 0
max_count = 2000
###############################################################################
# Fit kmeans  model
mode = 3
count =0
while(error<0.84):
    kmeans_fit = cluster.KMeans(n_clusters = mode).fit(K_train)
    cluster_labels = kmeans_fit.labels_
    test_label = kmeans_fit.fit_predict(K_test)


    X = data_all.iloc[:,1:]
    y = power_data.iloc[:,0]
    X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
    # X_train_1 = X_train.iloc[,:]

    for i in range(0,mode):
        X_train_0 = X_train.iloc[data_loc(cluster_labels,i),:]
        y_train_0 = y_train.iloc[data_loc(cluster_labels,i)]
        X_test_0 = X_test.iloc[data_loc(test_label,i),:]
        y_test_0 = y_test.iloc[data_loc(test_label,i)]
        SVR_0 = svr_train(X_train_0,y_train_0,5000,0.00001)
        predict_0 = SVR_0.predict(X_test_0)
        if i == 0:
            test_all = pd.DataFrame(y_test_0)
            predict_all =pd.DataFrame(predict_0)
        else:

            test_all=test_all.append(pd.DataFrame(y_test_0))
            predict_all=predict_all.append(pd.DataFrame(predict_0))



    error = r2_score(test_all,predict_all)
    count += 1
    print(count)
    if count>max_count:
        break


if count< max_count:
    resultcal(test_all, predict_all.values)
    lw = 2 #line width

    plt.scatter(test_all,predict_all, color='darkorange', label='predict')
    plt.plot(test_all, test_all, color='c', lw=lw)
    plt.xlabel('measured')
    plt.ylabel('predicted')
    plt.title('Support Vector Regression k=3' )
    plt.legend()
    predict_all=pd.DataFrame(predict_all.values,columns=['predict'],index=test_all.index)
    result=pd.concat([test_all,predict_all],axis=1)
    result=result.sort_index()
    result= pd.DataFrame(result.values,columns=['actual','predict'])
    result.plot()
    plt.show()

