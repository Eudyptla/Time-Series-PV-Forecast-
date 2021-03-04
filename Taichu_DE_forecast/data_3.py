import pandas as pd
import numpy as np

data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI.csv',encoding='utf-8-sig',index_col=0)
time_series = pd.date_range(data.index[0],data.index[-1],freq='H')
data.index = time_series

sun_list =[]
log = -1
for i in data.index:
    log = log+1
    if (i.hour >5) and (i.hour <19) :
        sun_list = sun_list+[log]

data_sun = data.iloc[sun_list,:]

hour = data_sun.index.hour
month = data_sun.index.month
date = data_sun.index.day
time = pd.DataFrame([month,date,hour],columns = data_sun.index,index = ['month','date','hour'])
time = time.T
data_sun =pd.concat([data_sun,time],axis=1)
print(data_sun)

data_sun.to_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun.csv',encoding='utf-8-sig')

