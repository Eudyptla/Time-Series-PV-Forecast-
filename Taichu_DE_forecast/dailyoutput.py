import numpy as np
import pandas as pd

data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun2.csv',encoding='utf-8-sig',index_col=0)

power = data.loc[:,'solarirrandance']


new_index = []
for i in power.index:
    i = pd.to_datetime(i)
    new_index += [i]

power.index =new_index
sun_list =[]
log = -1
for i in power.index:
    log = log + 1
    if (i.hour<18 and i.hour >7):
        sun_list +=  [log]


sun_data = power[sun_list].values
sun_data=sun_data.reshape(-1,10)
sun_data = pd.DataFrame(sun_data,columns=np.arange(8,18,1),index = pd.date_range('2016/01/1','2017/012/31',freq='D'))
sun_data = sun_data.dropna()
sun_data.to_csv('Taichung_DE_daily_Irandance.csv', encoding='utf-8-sig')




