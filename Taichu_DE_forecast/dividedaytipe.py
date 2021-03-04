import pandas as pd
import re
import numpy as np
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding= 'utf-8-sig',index_col=0)
cloud =data.loc[:,"Cloud Amount"]
cloud = cloud.dropna()
index_weather = np.array([])

for i in range(0,731):
    avg = sum(cloud.values[i*5:i*5+5])/5
    print(cloud.values[i*5:i*5+5])
    print(avg)
    if (avg > 7) :
        index_weather = np.append(index_weather,np.zeros((13,),dtype=int))

    else:
        index_weather = np.append(index_weather,np.ones((13,),dtype=int))



nom_index= np.nonzero(index_weather)
abnom_index =np.where(index_weather<1)

data.iloc[abnom_index[0],:].to_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_abnomsun.csv',encoding='utf-8-sig')
data.iloc[nom_index[0],:].to_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_nomsun.csv',encoding='utf-8-sig')




