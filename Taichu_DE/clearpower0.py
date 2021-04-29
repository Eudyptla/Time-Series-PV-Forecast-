import pandas as pd
import numpy as np
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun.csv',encoding='utf-8-sig',index_col=0)
power = data.loc[:,'power(KWH)']
index = np.where(power != 0)[0]
data2=data.iloc[index,:]

data2.to_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun2.csv',encoding='utf-8-sig')