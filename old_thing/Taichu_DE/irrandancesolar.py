import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding= 'utf-8-sig',index_col=0)
def create_lags(df, N):
    df = pd.DataFrame({'value': df})
    for i in N:
        df['Lag' + str(i+1)] = df.value.shift(i+1)
    return df

x = data.loc[:,'power(KWH)'].values
x_2 = data.loc[:,'solarirrandance'].values
x_2 = create_lags(x_2, np.arange(0,15))
data_1 = pd.DataFrame([x])
data_all = pd.concat([data_1.T,x_2],axis=1)
data_all=data_all.dropna()
ans_cor=[]
ans_cor1=[]
for i in data_all.columns[2:]:
    ans = np.corrcoef(data_all.iloc[:,0],data_all.loc[:,i])
    ans_cor += [ans[0,1]]
for i in range(2,len(data_all.columns)-1):
    ans = np.corrcoef(data_all.iloc[:,i+1],data_all.iloc[:,i])
    ans_cor1 += [ans[0,1]]

# print(ans_cor)

# print(len(ans_cor))
plt.stem(np.arange(0,len(ans_cor)),ans_cor)
print(ans_cor)
print(ans_cor1)
r0 = ans_cor[0]
pcf_ans=[r0]
for i in range(0,12):
    pcf=(ans_cor[i]-ans_cor[i+1]*ans_cor1[i])/((1-(ans_cor[i+1])**2)**0.5)/((1-(ans_cor1[i])**2)**0.5)
    pcf_ans += [pcf]

print(pcf_ans)
plt.figure()
plt.stem(np.arange(0,len(pcf_ans)),pcf_ans)


plt.show()