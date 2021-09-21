import pandas as pd
import numpy as np
data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding= 'utf-8-sig',index_col=0)
month =data.loc[:,"month"]
five_nine_index = []
else_index =[]
l = 0
for i in month.values:
    if ((i == 5 or i == 6) or (i == 7 or i ==8))or i == 9 :
        five_nine_index =five_nine_index+[l]
        l=l+1
    else:
        else_index =else_index+[l]
        l=l+1


five_nine_data = data.iloc[five_nine_index,:]
else_data = data.iloc[else_index,:]

five_nine_data.to_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun59.csv',encoding='utf-8-sig')
else_data.to_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sunelse.csv',encoding='utf-8-sig')

