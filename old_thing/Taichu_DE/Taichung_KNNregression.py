import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
import numpy as np
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding='utf-8-sig',index_col=0)

def create_lags(df, N):
    for i in N:
        df['Lag' + str(i+1)] = df.value.shift(i+1)
    return df

x = data.iloc[:,16]
x_2 = data.iloc[:,17]
temp = data.iloc[:,2]
RH = data.iloc[:,4]
wind_speed=data.iloc[:,5]
sunshine =data.iloc[:,11]

x = pd.DataFrame({'value':x})
x_2 = pd.DataFrame({'value':x_2})
x_2 = create_lags(x_2,[1,2])
x = create_lags(x,[0,1])
wind_speed = pd.DataFrame({'WS':wind_speed})
temp = pd.DataFrame({'temp':temp})
RH = pd.DataFrame({'RH':RH})
X = pd.concat([x,x_2.iloc[:,1:],temp,RH,wind_speed],axis=1)
X=X.dropna()
y=X.iloc[:,0]
X = X.iloc[:,1:]
train_idx = int(len(X) * .8)
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]

n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X_train, y_train).predict(X_test)
    if i == 0:
        y_uni = y_
    plt.subplot(2, 1, i + 1)
    plt.scatter(y_test,y_, c='k', label='data')
    plt.plot(y_test,y_test, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))

plt.tight_layout()
plt.show()

# result_uni=pd.DataFrame(y_test.values,columns=['actual power'],index=y_test.index)
# result_uni2 = pd.DataFrame(y_uni,columns=['predict power'],index=y_test.index)
# result_uni = pd.concat([result_uni,result_uni2],axis=1)
# print(result_uni)
# result_uni.plot()
# plt.show()
# error = mean_squared_error(y_test, y_uni)
# print('Test MSE: %.3f' % error)
# Root_error =error**0.5
# print('Test RMSE: %.3f' % Root_error)

result=pd.DataFrame(y_test.values,columns=['actual power'],index=y_test.index)
result_2 = pd.DataFrame(y_,columns=['predict power'],index=y_test.index)
result = pd.concat([result,result_2],axis=1)
print(result)
result.plot()
plt.show()
error = mean_squared_error(y_test, y_)
print('Test MSE: %.3f' % error)
Root_error =error**0.5
print('Test RMSE: %.3f' % Root_error)

