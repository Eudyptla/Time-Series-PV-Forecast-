import pandas as pd
data = pd.read_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_weather_8-17-5.csv',encoding='utf-8',index_col=0)

# hour = data.loc[:,'hour']
# min = data.loc[:,'minute']
# all_min =60*hour+min
#
# all_min = pd.DataFrame({'all min':all_min},index=data.index)
# result= pd.concat([data,all_min],axis=1)
new_index = []
for i in data.index:
    i = pd.to_datetime(i)
    new_index=new_index+[i]

new_index = pd.DataFrame([],index = new_index,columns=[])
# print(new_index.index)

time_col = pd.DataFrame([new_index.index.month,new_index.index.day],columns=data.index,index=['month','day'])
time_col = time_col.T

result = pd.concat([data,time_col],axis =1 )

result.to_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_weather_8-17-5.csv',encoding='utf-8-sig')
