import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_0 = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_power_weather.csv',encoding='utf-8-sig',index_col=0)
# x = data_0.iloc[:,17].values
# # x = data_0.iloc[:,12] #from weather station
# y = data_0.iloc[:,16]
# x = data_0.iloc[:-1,17].values #late 1 hour
# x = data_0.iloc[:-1,12] #from weather station late 1hour
# y = data_0.iloc[1:,16] #late 1 hour
x = data_0.iloc[:,0].values
y = data_0.iloc[:,16]
L= ~np.isnan(y)
L_2 = ~np.isnan(x)
L = L&L_2
y =y[L]
x=x[L]
ans = np.corrcoef(x, y)
print(ans)
plt.scatter(x, y)
plt.show()
x=pd.DataFrame(x,columns=['Rad'],index=y.index)
result=x.join(y)
result.plot()
plt.show()

