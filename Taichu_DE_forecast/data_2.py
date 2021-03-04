import pandas as pd


weather = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\TAICHUNG.csv',encoding='utf-8-sig')
solar = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE.csv',encoding='utf-8-sig')
weather = pd.DataFrame(weather.iloc[:,2:],index=solar.index)
print(solar.head(5))

data =solar.join(weather)
data.to_csv('Taichung_DE_2.csv', encoding='utf-8-sig')
