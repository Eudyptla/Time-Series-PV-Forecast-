import pandas as pd
import numpy as np

data = pd.read_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_weather_8-17-5.csv',encoding= 'utf-8-sig',index_col=0)

power = data.loc[:,'核三生水池'].values

index = np.where(power != 0)[0]
print(index)
data2=data.iloc[index,:]

data2.to_csv('E:\\Users\\Documents\\10minforecast\\nuclear3_sun_weather_8-17-6.csv',encoding='utf-8-sig')

