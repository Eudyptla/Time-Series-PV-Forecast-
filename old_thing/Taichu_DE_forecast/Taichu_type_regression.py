from sklearn import tree
import pandas as pd

data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_daily_type.csv',encoding='utf-8-sig',index_col=0)
# 切分訓練與測試資料

power = data.power
type =data.type
weather_data = data.iloc[:,2:15]
data =pd.concat([power,type,weather_data],axis=1)
data=data.dropna()
X = data.iloc[:,2:]
y = data.iloc[:,1]
train_idx = int(len(X) * .9)
# create train and test data
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]

# 建立分類器
clf = tree.DecisionTreeClassifier()
tree_clf = clf.fit(X_train,y_train)
#
# 預測
test_y_predicted = tree_clf.predict(X_test)
y_test2=y_test.values
y_test2 = y_test2.reshape(-1,1)
print(clf.score(X_test,y_test2))


# 標準答案

