from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
y = x.iloc[:, 0]
X = x.iloc[:, 1:]
train_idx = int(len(X) * .8)
# create train and test data
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
# fit and predict


train_stats = X_train.describe()
train_stats = train_stats.transpose()
print(train_stats)

train_labels = y_train
test_labels = y_test
print(test_labels)
#
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(X_train)
normed_test_data = norm(X_test)

def build_model():
  model = keras.Sequential([
    layers.Dense(16, activation=tf.nn.relu, input_shape=[len(X_train.keys())]),
    layers.Dense(16, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()

print(model.summary())
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
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$power^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')

    plt.legend()
    plt.show()
# The patience parameter is the amount of epochs to check for improvement
callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="auto")

history = model.fit(normed_train_data, train_labels, epochs=1000,
                    validation_split = 0.2, verbose=0, callbacks=[callback])

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)


test_predictions = model.predict(normed_test_data).flatten()

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
R_2 = r2_score(test_labels,test_predictions)

print("Testing set Mean Abs Error: {:5.2f} power".format(mae))
print("Testing set Mean Square Error: {:5.2f} power".format(mse))
print("Testing set R^2 : {:5.3f} power".format(R_2))
print("Testing set Root Mean Square Error: {:5.2f} power".format(mse**0.5))

plt.show()