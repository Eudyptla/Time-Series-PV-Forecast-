from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 1 準備數據
# 讀取波士頓地區房價信息
boston = load_boston()
# 查看數據描述
print(boston.DESCR)   # 共506條波士頓地區房價信息，每條13項數值特徵描述和目標房價
# 查看數據的差異情況
print("最大房價：", np.max(boston.target))   # 50
print("最小房價：",np.min(boston.target))    # 5
print("平均房價：", np.mean(boston.target))   # 22.532806324110677

x = boston.data
y = boston.target
print(x)
print(y)

# 2 分割訓練數據和測試數據
# 隨機採樣25%作為測試 75%作為訓練
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)


# 3 訓練數據和測試數據進行標準化處理
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 4.1 支持矢量機模型進行學習和預測
# 線性核函數配置支持矢量機
linear_svr = SVR(kernel="linear")
# 訓練
linear_svr.fit(x_train, y_train)
# 預測 保存預測結果
linear_svr_y_predict = linear_svr.predict(x_test)

# 多項式核函數配置支持矢量機
poly_svr = SVR(kernel="poly")
# 訓練
poly_svr.fit(x_train, y_train)
# 預測 保存預測結果
poly_svr_y_predict = linear_svr.predict(x_test)

# 5 模型評估
# 線性核函數 模型評估
print("線性核函數支持矢量機的默認評估值為：", linear_svr.score(x_test, y_test))
print("線性核函數支持矢量機的R_squared值為：", r2_score(y_test, linear_svr_y_predict))
print("線性核函數支持矢量機的均方誤差為:", mean_squared_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(linear_svr_y_predict)))
print("線性核函數支持矢量機的平均絕對誤差為:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                                 ss_y.inverse_transform(linear_svr_y_predict)))
# 對多項式核函數模型評估
print("對多項式核函數的默認評估值為：", poly_svr.score(x_test, y_test))
print("對多項式核函數的R_squared值為：", r2_score(y_test, poly_svr_y_predict))
print("對多項式核函數的均方誤差為:", mean_squared_error(ss_y.inverse_transform(y_test),
                                           ss_y.inverse_transform(poly_svr_y_predict)))
print("對多項式核函數的平均絕對誤差為:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(poly_svr_y_predict)))
