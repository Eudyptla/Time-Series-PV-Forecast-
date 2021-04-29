import pandas as pd
data = pd.read_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_weather_8-17-5.csv',encoding='utf-8',index_col=0)

month =data.loc[:,'month']
day =data.loc[:,'day']

def oriday(x):
    if x ==1:
        return 0
    elif x==2:
        return 31
    elif x==3:
        return 59
    elif x==4:
        return 90
    elif x==5:
        return 120
    elif x ==6:
        return 151
    elif x ==7 :
        return 181
    elif x==8:
        return 212
    elif x==9:
        return 243
    elif x==10:
        return 273
    elif x==11:
        return 304
    elif x==12:
        return 334
day_num=[]
for i in month:
    new = oriday(i)
    day_num = day_num+[new]

oridate = day+day_num

result = pd.concat([data,oridate],axis =1 )

result.to_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_weather_8-17-5.csv',encoding='utf-8-sig')

