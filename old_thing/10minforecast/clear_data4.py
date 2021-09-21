import numpy as np
import pandas as pd
import re
power_data = pd.read_csv('E:\\Users\\Documents\\10minforecast\\nuclear_3.csv',encoding='utf-8',index_col=0)
weather_data =pd.read_csv('E:\\Users\\Documents\\10minforecast\\HENGCHUN10min.csv',encoding='utf-8',index_col=0)
power_data=power_data.dropna()
weather_data =weather_data.dropna()
power_data = power_data[~power_data.index.duplicated(keep='first')]
weather_data = weather_data[~weather_data.index.duplicated(keep='first')]
time_series = pd.date_range('2018/09/24 12:50','2019/04/10 16:00',freq='10min')
l=0
j=0
data =pd.DataFrame([])
for i in time_series:
    i = str(i)
    if i.find(str(power_data.index[l]))>-1:
        power = power_data.iloc[l,0]
        l=l+1
    else:
        power =[]
    if i.find(str(weather_data.index[j]))>-1:
        weather = weather_data.iloc[j,:]
        j=j+1
    else:
        weather =[]


    data_new = pd.DataFrame(power, index={i}, columns={'核三生水池'})
    data_new2 = pd.DataFrame(weather, columns={i})

    data_new = pd.concat([data_new, data_new2.T], axis=1)
    data = data.append(data_new)
new_index=[]
for i in data.index:
    i = pd.to_datetime(i)
    new_index=new_index+[i]
# data = pd.DataFrame()
# print(data)
# i =str(time_series[0])
# if power_data.index[l].find(i) > -1:
#     power = power_data.iloc[l, 0]
#
# if i.find(str(weather_data.index[j]))>-1:
#     weather = weather_data.iloc[j,:]
#
# print(weather)
# data_new = pd.DataFrame(power,index = {i},columns={'核三生水池'})
# data_new2 = pd.DataFrame(weather,columns={i})
# print(data_new2)
# data_new = pd.concat([data_new,data_new2.T],axis = 1)
# data_new.index = {pd.to_datetime(i)}
# data = data.append(data_new)
#
# print(data)
# print(type(data.index))
# print(data.index.hour)

data.index=new_index
data.to_csv('E:\\Users\\Documents\\10minforecast\\nuclear_31.csv',encoding='utf-8-sig')