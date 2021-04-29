import numpy as np
import pandas as pd
import re
# power_data = pd.read_csv('E:\\Users\\Documents\\10minforecast\\solar.csv',encoding='utf-8',index_col=0)
weather_data =pd.read_csv('E:\\Users\\Documents\\10minforecast\\HENGCHUN10min.csv',encoding='utf-8',index_col=0)
# time = power_data.loc[:,'時間'].values
new_index=[]
for i in weather_data.index:
    if len(i) != 16:
        i = '2018/'+i

    new_index = new_index+[i]

weather_data.index = new_index
time_new =[]
# for s in time:
#     new = re.sub("-","/",str(s))
#     time_new = time_new+[new]
#
# num_0 = 0
# num_1 = 0

# for i in range(0,len(time_new)):
#     if time_new[i].find('2018/09/24 12:50') > -1:
#         num_0 = i
#         break
#
# for i in range(num_0,len(time_new)):
#     if time_new[i].find('2019/02/25 14:50') > -1:
#         num_1 = i
#         break

# nuclear_power = power_data.loc[:,'核三生水池'].values
# power_nuclear = pd.DataFrame(nuclear_power,index =time_new,columns={'核三生水池'})

# power_nuclear.to_csv('E:\\Users\\Documents\\10minforecast\\nuclear_3.csv',encoding='utf-8-sig')
weather_data.to_csv('E:\\Users\\Documents\\10minforecast\\HENGCHUN10min.csv',encoding='utf-8-sig')

