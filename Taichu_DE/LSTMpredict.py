import pandas as pd
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
####Function#############################################################################################################
def create_lags(df, N):
    for i in N:
        df['Lag' + str(i+1)] = df.value.shift(i+1)
    return df
####Function#############################################################################################################
def MAPE(x,y):
    n = len(x)

    d = (sum(abs((x-y)/x)))/n
    return d
####Function#############################################################################################################
def MRE(x,y):
    n=len(x)

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
####Main#############################################################################################################
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',
                          encoding= 'utf-8-sig',index_col=0)
hour=data.loc[:,'hour']


x = data.loc[:,'power(KWH)']
x_2 = data.loc[:,'solarirrandance']
x_2 = pd.DataFrame({'value':x_2})
x_2 = create_lags(x_2,[1,2])
x = pd.DataFrame({'value':x})
x = create_lags(x, [0,1])

temp = data.loc[:,'Temperaure']
RH = data.loc[:,'RH']
wind_speed=data.loc[:,'WS']
sunshine =data.loc[:,'SunShine']
GloblRad = data.loc[:,'GloblRad']
# cloud_cover = data.iloc[:,15]

date =data.loc[:,'date']
month = data.loc[:,'month']

x = pd.concat([x,x_2.iloc[:,1:],temp,RH,wind_speed,sunshine,GloblRad,hour,date,month],axis=1)
x = x.dropna()
x.index = np.arange(len(x))

#Normalization
train_stats = x.describe()
train_stats = train_stats.transpose()

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
train_norm = norm(x)


def buildTrain(train, pasttime, futuretime):
  X_train, Y_train = [], []
  for i in range(train.shape[0]-futuretime-pasttime):
    X_train.append(np.array(train.iloc[i:i+pasttime]))
    Y_train.append(np.array(train.iloc[i+pasttime:i+pasttime+futuretime,0]))
  return np.array(X_train), np.array(Y_train)
X_train, Y_train = buildTrain(train_norm, 1, 1)

# create train and test data
def splitData(X,Y,rate):
  X_train = X[:int(X.shape[0]*rate)]
  Y_train = Y[:int(Y.shape[0]*rate)]
  X_val = X[int(X.shape[0]*rate):]
  Y_val = Y[int(Y.shape[0]*rate):]
  return X_train, Y_train, X_val, Y_val

X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.9)

# fit and solve

def buildManyToOneModel(shape):

    model = Sequential()

    model.add(LSTM(8, input_length=shape[1], input_dim=shape[2]))
  # output shape: (1, 1)
    model.add(Dense(1))

    model.compile(loss="mean_squared_error", optimizer="adam"
                ,metrics=['mean_absolute_error', 'mean_squared_error'])
    model.summary()
    return model

model = buildManyToOneModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
history = model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])

print(model.summary())
# #
#
# Display training progress by printing a single dot for each completed epoch



def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [power]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    # plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$power^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    # plt.ylim([0, 20])
    plt.legend()

# The patience parameter is the amount of epochs to check for improvement

plot_history(history)


def return_value(Y):
    return Y*train_stats['std']['value']+train_stats['mean']['value']

Y_test_ture = return_value(Y_val)
test_predictions = model.predict(X_val).flatten()
test_value = return_value(test_predictions)
plt.figure()
plt.scatter(Y_test_ture, test_value,color='darkorange',label='predict')
plt.plot(Y_test_ture, Y_test_ture, color='c', lw=2)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('DRNN LSTM')
plt.legend()



result=pd.DataFrame(Y_test_ture,columns=['actual '])

result_2 = pd.DataFrame(test_value,columns=['predict '])
result_all = pd.concat([result, result_2], axis=1)
result_all.plot()
resultcal(result.values,result_2.values)
plt.show()

