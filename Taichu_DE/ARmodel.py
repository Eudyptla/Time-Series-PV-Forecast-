import pandas as pd
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from pandas import concat
from pandas.plotting import lag_plot
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf

data = pd.Series.from_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE.csv',header=0)
plot_acf(data, lags=31)
pyplot.show()