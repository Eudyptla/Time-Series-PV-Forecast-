from bs4 import BeautifulSoup
import requests
import pandas as pd
# data from https://www.taipower.com.tw/d006/loadGraph/loadGraph/genshx_.html
# analysis html get url
url = "https://www.taipower.com.tw/d006/loadGraph/loadGraph/data/genary.txt"
page = requests.get(url)
page.encoding='utf-8'
soup = BeautifulSoup(page.text,'lxml')
# get text
text = soup.get_text()
text_list = text.split('[')
L = len(text_list)

solar_p=[]
for i in range(1,L):
    if '太陽能' in text_list[i]:
        if '小計' not in text_list[i]:
            if '太陽能購電' not in text_list[i]:
             solar_p =solar_p + [i]
solar =[]
solar_index=[]
for i in solar_p:
    s = text_list[i]
    s = s.rstrip(', "],')
    s = s.split(',')
    for j in range(0,len(s)):
        s[j] = s[j].strip(' " ')
        s[j] = s[j].rstrip(' " ')

    solar = solar + [[s[3]]]
    solar_index = solar_index + [s[1]]
time =text_list[0]
str = ''
time = str.join(list(filter(lambda ch: ch in '0123456789-: ', time)))
time = time.strip(':')
time = [time.rstrip(':')]
#time = [time.rstrip(':')]
my_df =pd.read_csv('data\\solar.csv',encoding='utf-8',index_col=0)
solar = pd.DataFrame(solar,index=solar_index)
time  = pd.DataFrame(time,index=['時間'])
solar = time.append(solar)
solar = solar.T
my_df = my_df.append(solar,ignore_index = True)
my_df.to_csv('data\\solar.csv',encoding='utf-8-sig')


