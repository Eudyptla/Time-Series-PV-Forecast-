import pandas as pd
import numpy as np

data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding='utf-8-sig',index_col=0)
cloud =data.loc[:,'Cloud Amount']
cloud=cloud.dropna()

tag=0
cloud_type =[]
for i in cloud.values:
    if tag ==0:
        cloud_type += [i]
        tag += 1

    elif tag == 4:
        cloud_type += [i]
        cloud_type += [i]
        cloud_type += [i]
        tag = 0

    else :
        cloud_type += [i]
        cloud_type += [i]
        tag += 1
weather_type=[]
for i in cloud_type:
    if i <5:
        weather_type += [0]
    elif i>8:
        weather_type += [2]
    else:
        weather_type += [1]

sun_index =[]
tag = 0
for i in data.hour.values:
    if (i>7) and (i<18):
       sun_index += [tag]

    tag += 1
power = data.iloc[sun_index,[16,17,20]]
weather_type = pd.DataFrame(weather_type,columns = ['weather_type'],index = power.index)
data_result = pd.concat([power,weather_type],axis=1)
print(data_result)
data_result =data_result.dropna()
data_result.to_csv('Taichung_DE_cloudtype.csv', encoding='utf-8-sig')
