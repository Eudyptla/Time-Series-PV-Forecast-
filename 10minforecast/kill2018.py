import pandas as pd
weather_data =pd.read_csv('E:\\Users\\Documents\\10minforecast\\HENGCHUN10min.csv',encoding='utf-8',index_col=0)
index =[]
for i in weather_data.index:
    i = str(i)
    i = i.replace('2018/','')
    index = index +[i]

weather_data.index = index

weather_data.to_csv('E:\\Users\\Documents\\10minforecast\\HENGCHUN10min.csv',encoding='utf-8-sig')