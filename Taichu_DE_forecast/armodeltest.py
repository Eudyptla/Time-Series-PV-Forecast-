import pandas as pd
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_2.csv',encoding='utf-8-sig',index_col=0)
taichu_2 = data.iloc[6:18,0:2]
j=1
for i in range(1,45):
    while(sum(data.iloc[(6+j*24):(18+j*24),13])<9):
        j=j+1
        if j==37:
            j=j+1

    taichu_2 = taichu_2.append(data.iloc[(6+j*24):(18+j*24),0:2])
    j=j+1
taichu_3=taichu_2.iloc[:,1]
taichu_3=taichu_3.fillna(0)
date = taichu_2.iloc[:,0]
series = pd.Series(taichu_3.values,index=date.values)

series.plot()
pyplot.show()


from pandas.plotting import lag_plot
lag_plot(series)
pyplot.show()

from pandas import concat
values = pd.DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)
pyplot.show()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=31)
pyplot.show()

X = dataframe.values

train, test = X[1:len(X) - 60], X[len(X) - 60:]
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]


# persistence model
def model_persistence(x):
    return x


# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
# plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()


X = series.values
train, test = X[1:len(X)- 60], X[len(X)- 60 :]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()