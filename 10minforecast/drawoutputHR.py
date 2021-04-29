import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


data = pd.read_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_weather_8-17-5.csv',encoding= 'utf-8-sig',index_col=0)
min = data.loc[:,'minute'].values
zero_index = np.where(min<10)
x = data.loc[:,'核三生水池']
x.plot()
plt.figure()
x_2 = data.iloc[zero_index[0],2]
x_2.plot()

# result = pd.concat([x,x_2],axis=1)
# result = pd.DataFrame(result.values)
# result.plot()
plt.show()
