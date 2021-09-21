import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_2.csv',encoding='utf-8-sig',index_col=0)
taichu_2 = data.iloc[:,0:2]
taichu_3=taichu_2.iloc[:,1]
taichu_3 = taichu_3.fillna(0)
date = taichu_2.iloc[:,0]
series = pd.Series(taichu_3.values,index=date.values)

series.plot()
pyplot.show()
#print(series)

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

X = series.values
train_size = int(len(X) -240)
test_size = len(X) - train_size

train, test = X[0:train_size], X[train_size:len(X)]

# train autoregression
model = ARIMA(train,order=(6,1,2))
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
print(model_fit.summary())