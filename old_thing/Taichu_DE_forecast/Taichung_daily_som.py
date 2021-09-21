from minisom import MiniSom
from matplotlib.gridspec import GridSpec

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('E:\\Users\\Documents\\Taichu_DE\\Taichung_DE_daily.csv',encoding='utf-8-sig',index_col=0)

train_data = data.values

som = MiniSom(8, 8, train_data.shape[1], sigma=2., learning_rate=0.5,
              neighborhood_function='gaussian', random_seed=10)
som.pca_weights_init(train_data)
print("Training...")
som.train_batch(train_data, 1000, verbose=True)  # random training
print("\n...ready!")

win_map = som.win_map(train_data)

print(win_map.keys())
print(som.winner(train_data[1]))



