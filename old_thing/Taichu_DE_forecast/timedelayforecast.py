import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import time

data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_power_weather.csv',encoding='utf-8-sig',index_col=0)
power = data['power(KWH)']
power = power.fillna(0)

# define function for create N lags
def create_lags(data, N):
    for i in range(N):
        df = data.shift(i+1)
    return df
# create 10 lags
df = create_lags(power,1)
df = df.dropna()
# create X and y
y = df.values
X = power[1:].values
y = y.reshape(-1,1)
X = X.reshape(-1,1)
train_idx = int(len(X) * .8)
# create train and test data
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
# fit and predict
clf = SVR()
t0 = time.time()
svr_fit = time.time() - t0
clf.fit(X_train, y_train)
y_svr=clf.predict(X_test)
svr_predict = time.time() - t0
plt.scatter(X_test,y_test , c='k', label='data', zorder=1)
plt.hold('on')
plt.plot(X_test, y_svr, c='r',
         label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus Kernel Ridge')
plt.legend()
plt.show()
y_pre = y_svr.reshape(-1,1)
test_index = data.index
test_index = test_index[(train_idx+1):]
result=pd.DataFrame(y_test,columns=['actual power'],index=test_index)
result_2 = pd.DataFrame(y_pre,columns=['predict power'],index=test_index)
result = pd.concat([result,result_2],axis=1)
result.plot()
plt.show()
print(result)

