import pandas as pd
import re
import math
data = pd.read_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_weather_8-17-2.csv',encoding='utf-8',index_col=0)
wind = data.loc[:,'風力']

p = re.compile(r'\|')
wind_speed_all = []
wind_level_all = []
for i in wind:
    if type(i) == str:
        wind_1 = p.split(i)
        wind_speed = wind_1[0]
        wind_level = wind_1[1]

    else:
        wind_speed = []
        wind_level = []

    wind_speed_all = wind_speed_all+[wind_speed]
    wind_level_all = wind_level_all+[wind_level]


wind_all = pd.DataFrame({'風速':wind_speed_all,'蒲福風級':wind_level_all},index=data.index)
result= pd.concat([data,wind_all],axis=1)

# wind_1 = p.split(wind[0])
# print(wind_1)
# print(p.split(wind[0]))

result.to_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_weather_8-17-1.csv',encoding='utf-8-sig')
