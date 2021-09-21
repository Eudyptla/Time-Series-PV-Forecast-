import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_daily.csv',encoding='utf-8-sig',index_col=0)

fig,axes = plt.subplots()
data.boxplot()
plt.show()
