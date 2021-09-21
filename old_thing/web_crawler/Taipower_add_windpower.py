from bs4 import BeautifulSoup
import requests
import pandas as pd
# data from https://www.taipower.com.tw/d006/loadGraph/loadGraph/genshx_.html
# analysis html construct get url
url = "https://www.taipower.com.tw/d006/loadGraph/loadGraph/data/genary.txt"
page = requests.get(url)
page.encoding = 'utf-8'
soup = BeautifulSoup(page.text, 'lxml')
# get text
text = soup.get_text()
text_list = text.split('[')
# clean useless data and get wind power
wind_temp = [i for i in text_list if ('風力' in i) & ('小計' not in i)]
wind_temp = [i.rstrip(', "],').split(',') for i in wind_temp]
wind = {i[1].strip(' " ').rstrip(' " '): [float(i[3].strip(' " ').rstrip(' " '))] for i in wind_temp}
# get time
time = text_list[0]
time_str = ''
time = time_str.join(list(filter(lambda ch: ch in '0123456789-: ', time)))
time = time.strip(':').rstrip(':')
# save data
my_df = pd.read_csv('data\\wind.csv', encoding='utf-8', index_col=0)
wind = pd.DataFrame(wind)
wind['時間'] = time
my_df = my_df.append(wind, ignore_index=True)
first_col = my_df.pop('時間')  # set time as first col
my_df.insert(0, '時間', first_col)
my_df.to_csv('data\\wind.csv', encoding='utf-8-sig')


