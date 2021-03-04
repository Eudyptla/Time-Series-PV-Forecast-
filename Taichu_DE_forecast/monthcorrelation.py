import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding= 'utf-8-sig',index_col=0)
def cal_cor(x,y):
    L= ~np.isnan(y)
    L_2 = ~np.isnan(x)
    L = L&L_2
    y =y[L]
    x=x[L]
    ans = np.corrcoef(x, y)
    return ans[0,1]

def month_index(x,y):
    ans =[]
    tag = 0
    for i in y:
        if i == x:
            ans = ans+[tag]
        tag=tag+1
    return ans
correlation_data = pd.DataFrame(columns=data.columns[:-3])
for i in range(1,13):
    month_dex = month_index(i,data.loc[:,'monh'])
    temp = []
    for j in data.columns[:-3]:
        data_1 = data.iloc[month_dex,:]
        correlation = cal_cor(data_1.loc[:,'power(KWH)'],data_1.loc[:,j])
        temp=temp+[correlation]
    temp = pd.DataFrame(temp,index= data.columns[:-3],columns=[i])
    temp =temp.T
    correlation_data =correlation_data.append(temp)

print(correlation_data)
for i in correlation_data.columns:
    plt.figure()
    correlation_data.loc[:,i].plot(kind ='bar',rot=0)

plt.show()




