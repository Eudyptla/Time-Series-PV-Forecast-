import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('E:\\Users\\Documents\\10minforecast\\nuclear_31.csv',encoding='utf-8',index_col=0)
data = data.dropna()
new_index = []
for i in data.index:
    i = pd.to_datetime(i)
    new_index=new_index+[i]

new_index = pd.DataFrame([],index = new_index,columns=[])
# print(new_index.index)

time_col = pd.DataFrame([new_index.index.hour,new_index.index.minute],columns=data.index,index=['hour','minute'])
time_col = time_col.T

data = pd.concat([data,time_col],axis =1 )


sun_list =[]
log = -1
for i in new_index.index:
    log = log+1
    if (i.hour >7) and (i.hour <18) :
        sun_list = sun_list+[log]

sun_data = data.iloc[sun_list,:]
# sun_power =sun_data.loc[:,'核三生水池']
# sun_power = pd.DataFrame(sun_power.values,columns={'Power'},index = sun_power.index)
# sun_data.plot()
# sun_power.plot()
# plt.show()
# sun_power.to_csv('E:\\Users\\Documents\\10minforecast\\nuclear_3_sun.csv',encoding='utf-8-sig')
# data.to_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_weather_time.csv',encoding='utf-8-sig')
# sun_power.to_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_clock.csv',encoding='utf-8-sig')
sun_data.to_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_weather_8-17.csv',encoding='utf-8-sig')