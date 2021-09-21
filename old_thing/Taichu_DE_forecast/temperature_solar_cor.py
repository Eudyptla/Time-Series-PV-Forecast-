import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


data_0 = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_power_weather.csv',encoding='utf-8-sig',index_col=0)
x = data_0.iloc[:,16]
x_2 = data_0.iloc[:,17]
temp = data_0.iloc[:,2]
RH = data_0.iloc[:,4]
UVI = data_0.iloc[:,14]
sunshine =data_0.iloc[:,11]
cloud_cover = data_0.iloc[:,15]
wind_speed=data_0.iloc[:,5]
x = pd.DataFrame({'value':x})
x_2 = pd.DataFrame({'value':x_2})
temp = pd.DataFrame({'temp':temp})
RH = pd.DataFrame({'RH':RH})
sunshine=pd.DataFrame({'Sunshine':sunshine})
cloud_cover = pd.DataFrame({'Cloud Amoun':cloud_cover})
wind_speed = pd.DataFrame({'WS':wind_speed})
UVI = pd.DataFrame({'UVI':UVI})
hour=pd.Series(np.array(range(1,len(x)+1))%24)
hour = pd.DataFrame({'Hour':hour.values},index=data_0.index)
def create_lags(df, N):
    for i in N:
        df['Lag' + str(i+1)] = df.value.shift(i+1)
    return df

x = create_lags(x,[0,1])
x_2 = create_lags(x_2,[0])
x = pd.concat([x,x_2.iloc[:,1:],temp,RH,wind_speed,hour,cloud_cover,UVI,sunshine],axis=1)
# x = pd.concat([x,temp,RH,wind_speed,hour],axis=1)
x = x.dropna()
output =x.iloc[:,0].values
solar_t_1 = x.iloc[:,1]
solar_t_2 = x.iloc[:,2]
solar_ira = x.iloc[:,3]
temperaure =x.loc[:,'temp'].values
RH=x.loc[:,'RH'].values
wind_speed = x.loc[:,'WS'].values
hour= x.loc[:,'Hour'].values
cloud_cover= x.loc[:,'Cloud Amoun'].values
sunshine = x.loc[:,'Sunshine'].values
UVI=x.loc[:,'UVI'].values

solar_t1_cor = np.corrcoef(solar_t_1, output)
solar_t2_cor = np.corrcoef(solar_t_2, output)
solar_ira_cor = np.corrcoef(solar_ira, output)
temp_cor = np.corrcoef(temperaure, output)
RH_cor = np.corrcoef(RH, output)
WS_cor = np.corrcoef(wind_speed, output)
Hour_cor = np.corrcoef(hour,output)
Cloud_cor = np.corrcoef(cloud_cover,output)
sunshine_cor =np.corrcoef(sunshine,output)
UVI_cor =np.corrcoef(UVI,output)

correlation = [solar_t1_cor.item(0,1),solar_t2_cor.item(0,1),solar_ira_cor.item(0,1),
               temp_cor.item(0, 1),RH_cor.item(0, 1),WS_cor.item(0,1),Hour_cor.item(0,1),Cloud_cor.item(0,1),
               sunshine_cor.item(0, 1),UVI_cor.item(0,1)]

print(correlation)
name=['T-1','T-2','Solar Irradiance','temperature','Relative humidity','Wind speed','time','Cloud cover','sunshine hour','UVI']

plt.bar(name,correlation)
plt.show()
