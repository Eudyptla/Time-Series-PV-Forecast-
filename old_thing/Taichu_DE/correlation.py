import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding= 'utf-8-sig',index_col=0)
def cal_cor(x,y):
    L= ~np.isnan(y)
    L_2 = ~np.isnan(x)
    L = L&L_2
    y =y[L]
    x=x[L]
    ans = np.corrcoef(x, y)
    return ans[0,1]

def create_lags(df, N):
    df = pd.DataFrame({'value': df})
    for i in N:
        df['Lag' + str(i+1)] = df.value.shift(i+1)
    return df
x = data.loc[:,'power(KWH)'].values
cor_all =[]
cor_name =[]
cor_all2 =[]
cor_name2 =[]

for i in data.columns[[0,1,2,3,4,5]]:
    cor=cal_cor(x,data.loc[:,i].values)
    cor_name += [i]
    cor_all += [cor]

for i in data.columns[[7,9,10,11,12]]:
    cor=cal_cor(x,data.loc[:,i].values)
    cor_name2 += [i]
    cor_all2 += [cor]


corplt = pd.Series(cor_all,index=cor_name)
corplt.plot(kind ='bar',rot=0)
plt.figure()
corplt2 = pd.Series(cor_all2,index=cor_name2)
corplt2.plot(kind ='bar',rot=0)
# plt.figure()
# autocorrelation_plot(power)
# plot_acf(power.values, lags=12)
# plot_pacf(power.values, lags=12)
# print(corplt)
#
# plt.stem(cor_name,cor_all)
plt.show()