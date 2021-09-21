import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from hpelm import ELM
raw_dataset = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',
                          encoding= 'utf-8-sig',index_col=0)

data = raw_dataset.copy()

x = data.loc[:,'power(KWH)']
x_2 = data.loc[:,'solarirrandance']
temp = data.loc[:,'emperaure']
RH = data.loc[:,'RH']
wind_speed=data.loc[:,'WS']
sunshine =data.loc[:,'SunShine']
GloblRad = data.loc[:,'GloblRad']
# cloud_cover = data.iloc[:,15]
hour=data.loc[:,'hour']
date =data.loc[:,'dae']
month = data.loc[:,'monh']
x = pd.DataFrame({'value':x})
x_2 = pd.DataFrame({'value':x_2})

def create_lags(df, N):
    for i in N:
        df['Lag' + str(i+1)] = df.value.shift(i+1)
    return df
x = create_lags(x,[0,1])
x_2 = create_lags(x_2,[0])

x = pd.concat([x,x_2.iloc[:,1:],temp,RH,wind_speed,sunshine,GloblRad,hour,date,month],axis=1)
x = x.dropna()
x.index = np.arange(len(x))
y = x.iloc[:, 0]
X = x.iloc[:, 1:]
train_idx = int(len(X) * .8)
# create train and test data
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]

#Normalization
train_stats = X_train.describe()
train_stats = train_stats.transpose()
train_labels = y_train
test_labels = y_test

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(X_train)
normed_test_data = norm(X_test)
# fit and predict

elm = ELM(X_train.shape[1], 1)
elm.add_neurons(10, "sigm")
elm.add_neurons(20, "rbf_l2")
elm.train(X_train.values, y_train.values, "LOO")
test_predictions = elm.predict(X_test.values)


plt.scatter(test_labels, test_predictions,color='darkorange')
plt.plot(y_test.values, y_test.values, color='c', lw=2)
plt.xlabel('True Values [power]')
plt.ylabel('Predictions [power]')
plt.axis('equal')
plt.axis('square')

result=pd.DataFrame(test_labels.values,columns=['actual power'],index=test_labels.index)

result_2 = pd.DataFrame(test_predictions,columns=['predict power'],index=test_labels.index)
result = pd.concat([result,result_2],axis=1)
print(result)
result.plot()
mae = mean_absolute_error(test_labels,test_predictions)
mse = mean_squared_error(test_labels,test_predictions)
R_2 = r2_score(test_labels,test_predictions)

print("Testing set Mean Abs Error: {:5.2f} power".format(mae))
print("Testing set Mean Square Error: {:5.2f} power".format(mse))
print("Testing set R^2 : {:5.3f} power".format(R_2))
print("Testing set Mean Square Error: {:5.2f} power".format(mse**0.5))

plt.show()
