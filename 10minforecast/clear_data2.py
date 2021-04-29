import numpy as np
import pandas as pd
import re
power_data = pd.read_csv('E:\\Users\\Documents\\10minforecast\\nuclear_3.csv',encoding='utf-8',index_col=0)
weather_data =pd.read_csv('E:\\Users\\Documents\\10minforecast\\HENGCHUN10min.csv',encoding='utf-8',index_col=0)

num_0 = 0
num_1 = 0
time_new = power_data.index
l=-1
for i in time_new:
    i=str(i)
    l=l+1
    if i.find('2018/09/24 12:50') > -1:
        num_0 = l
        break

for i in time_new[num_0:]:
    i = str(i)
    l=l+1
    if i.find('2019/04/10 16:00') > -1:
        num_1 = l
        break

time=[]
for i in time_new[num_0:num_1]:
    i = pd.to_datetime(i)
    time =time+[i]
weather_time = []
for i in weather_data.index:
    i = pd.to_datetime(i)
    weather_time =weather_time + [i]

weather_data.index = weather_time


nuclear_3 = pd.DataFrame(power_data.iloc[num_0:num_1,0],index = time)

nuclear_3.to_csv('E:\\Users\\Documents\\10minforecast\\nuclear_3.csv',encoding='utf-8-sig')
weather_data.to_csv('E:\\Users\\Documents\\10minforecast\\HENGCHUN10min.csv',encoding='utf-8-sig')




