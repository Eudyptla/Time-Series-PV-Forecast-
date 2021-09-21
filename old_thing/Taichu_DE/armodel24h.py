import pandas as pd
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding='utf-8-sig',index_col=0)
power = data.loc[:,'power(KWH)'].dropna()
series = pd.Series(power.values)
series.plot()
pyplot.figure()
from pandas.plotting import lag_plot
lag_plot(series)
pyplot.figure()

from pandas import concat
values = pd.DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=15)

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(series, lags=15)
pyplot.figure()

X = series.values
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
train, test = X[:train_size], X[train_size:]
# train autoregression
model = AR(train)
model_fit = model.fit(maxlag=10,maxiter=100,tol=1e-5)
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=True)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

print(model_fit.information)

