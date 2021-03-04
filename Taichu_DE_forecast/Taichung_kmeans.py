from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster


data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun2.csv',encoding='utf-8-sig',index_col=0)
temp = data.loc[:,'Temperature']
RH = data.loc[:,'RH']
wind_speed=data.loc[:,'WS']
sunshine =data.loc[:,'SunShine']
# cloud_cover = data.iloc[:,15]
hour=data.loc[:,'hour']
date =data.loc[:,'date']
month = data.loc[:,'month']


X = pd.concat([temp,RH,wind_speed,sunshine,hour,date,month],axis = 1)
X =X.dropna()

train_idx = int(len(X) * .8)

# create train and test data
X_train, X_test, = X[:train_idx], X[train_idx:]


# 迴圈
silhouette_avgs = []
ks = range(2, 10)
for k in ks:
    kmeans_fit = cluster.KMeans(n_clusters = k).fit(X_train)
    cluster_labels = kmeans_fit.labels_
    silhouette_avg = metrics.silhouette_score(X_train, cluster_labels)
    silhouette_avgs.append(silhouette_avg)

# 作圖並印出 k = 2 到 10 的績效
plt.bar(ks, silhouette_avgs)
plt.show()
print(silhouette_avgs)
