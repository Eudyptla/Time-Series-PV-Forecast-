import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


data = pd.read_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_weather_8-17-5.csv',encoding= 'utf-8-sig',index_col=0)
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

x = data.loc[:,'核三生水池'].values

tem_c = data.loc[:,'溫度']
tem_f = data.loc[:,'溫度.1']
RH = data.loc[:,'相對溼度']
hour = data.loc[:,'hour']
minute = data.loc[:,'minute']
sunhour = data.loc[:,'日照時數']
allmin = data.loc[:,'all min']
month =data.loc[:,'month']
day =data.loc[:,'day']
oriday =data.loc[:,'day2']
wind_level = data.loc[:,'蒲福風級']
wind_speed = data.loc[:,'風速']
xtem_c_cor = cal_cor(x,tem_c)
xtem_f_cor = cal_cor(x,tem_f)
xRH_cor = cal_cor(x,RH)
xhour_cor = cal_cor(x,hour)
xminute_cor = cal_cor(x,minute)
xsunhour_cor = cal_cor(x,sunhour)
xallmin_cor = cal_cor(x,allmin)
xwind_spcor =cal_cor(x,wind_speed)
xwind_levelcor =cal_cor(x,wind_level)
xmonth_cor = cal_cor(x,month)
xday_cor =cal_cor(x,day)
xday2_cor=cal_cor(x,oriday)
correlation =  [xtem_c_cor,xtem_f_cor,xRH_cor,xhour_cor,xminute_cor,xsunhour_cor,xallmin_cor,xwind_spcor,xwind_levelcor,
                xmonth_cor,xday_cor,xday2_cor]
cor_name=['tem_C','tem_F','RH','Hour','Minute','Sunhour','Def_time','Wind_speed','Wind_level','Month','Date','Orinalday']

plt.bar(cor_name,correlation)


x_1 = create_lags(x,[0,1,2,3,4,5,6,7,8,9])
#
# xx_1_cor = cal_cor(x,x_1.loc[:,'Lag1'])
# xx_2_cor = cal_cor(x,x_1.loc[:,'Lag2'])
# xx_3_cor = cal_cor(x,x_1.loc[:,'Lag3'])
# xx_4_cor = cal_cor(x,x_1.loc[:,'Lag4'])
# xx_5_cor = cal_cor(x,x_1.loc[:,'Lag5'])
# xx_6_cor = cal_cor(x,x_1.loc[:,'Lag6'])
# xx_7_cor = cal_cor(x,x_1.loc[:,'Lag7'])
# xx_8_cor = cal_cor(x,x_1.loc[:,'Lag8'])
# xx_9_cor = cal_cor(x,x_1.loc[:,'Lag9'])
# xx_10_cor = cal_cor(x,x_1.loc[:,'Lag10'])
#
# print(xx_1_cor)
# print(xx_2_cor)
# print(xx_3_cor)
# print(xx_4_cor)
# print(xx_5_cor)
# print(xx_6_cor)
# print(xx_7_cor)
# print(xx_8_cor)
# print(xx_9_cor)
# print(xx_10_cor)
# print(xtem_c_cor)
plt.figure()
autocorrelation_plot(x)
plot_acf(x, lags=10)
plot_pacf(x,lags=10)
plt.show()
# plt.scatter(x, y)
# plt.show()
# result=pd.DataFrame([x,y.values],columns= y.index,index=['power','temperature'])
# result = result.T
# result.plot()
# plt.show()
