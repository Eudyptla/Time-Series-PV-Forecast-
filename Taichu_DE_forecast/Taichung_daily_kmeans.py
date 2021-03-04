from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn import cluster


data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_daily.csv',encoding='utf-8-sig',index_col=0)
X =data.dropna()

train_idx = int(len(X) * .8)

# create train and test data
X_train, X_test, = X[:train_idx], X[train_idx:]


# 迴圈
# silhouette_avgs = []
# ks = range(2, 10)
# for k in ks:
#     kmeans_fit = cluster.KMeans(n_clusters = k).fit(X_train)
#     cluster_labels = kmeans_fit.labels_
#     silhouette_avg = metrics.silhouette_score(X_train, cluster_labels)
#     silhouette_avgs.append(silhouette_avg)
#
# # 作圖並印出 k = 2 到 10 的績效
# plt.bar(ks, silhouette_avgs)
# plt.show()
# print(silhouette_avgs)
kmeans_fit = cluster.KMeans(n_clusters = 4).fit(X)
cluster_labels = kmeans_fit.labels_
out_label = pd.DataFrame(cluster_labels,index = X.index,columns={'kind'})
data_out = pd.concat([X,out_label],axis =1)
data_2 = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_WUQI_sun4.csv',encoding='utf-8-sig',index_col=0)
data_out.columns = ['08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','kind']
data_new =pd.DataFrame()
for i in data_out.index:
    for j in data_out.columns[:-1]:
        power = data_out.loc[i,j]
        type = data_out.loc[i,'kind']
        data_0 = pd.DataFrame([power,type],columns = [pd.to_datetime(i+'  '+j)],index=['power','type'])
        data_0=data_0.T
        data_new = data_new.append(data_0)



new_index = []
for i in data_2.index:
    i = pd.to_datetime(i)
    new_index=new_index+[i]

data_2.index = new_index
data_out2 = pd.concat([data_new,data_2],axis =1)
data_out2.to_csv('Taichung_DE_daily_type.csv', encoding='utf-8-sig')