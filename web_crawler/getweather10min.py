from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np
import pandas as pd
my_df =pd.read_csv('E:\\Users\\Documents\\10minweather\\HENGCHUN10min.csv',encoding='utf-8',index_col=0)

url = "https://www.cwb.gov.tw/V7/observe/24real/Data/46759.htm"
html = urlopen(url).read()
soup = BeautifulSoup(html, "lxml")
data_tag = soup.find(class_='BoxTable')
text_list = data_tag.get_text('<tr>')
text =text_list.split('<tr>')
def kill_n(n):
    return n != '\n'

text_2 = filter(kill_n,text)
text = list(text_2)
text[3]='溫度.1'
text.remove('(°C>>°F)')
text.remove('(°F>>°C)')
text.remove('(m/s) | (級)')
text.remove('(m/s) | (級)')
text.remove('(公里)')
text.remove('(%)')
text.remove('(百帕)')
text.remove('雨量(毫米)')
text.remove('(小時)')
text=np.array(text)
text=text.reshape((-1,12))
text=pd.DataFrame(text[1:,1:],index = text[1:,0],columns=text[0,1:])
text=text.sort_index()
next_tag=my_df.index[-1]
j = text.index.get_loc(next_tag)
new_text = text.iloc[j+1:,:]
my_df = my_df.append(new_text)

my_df.to_csv('E:\\Users\\Documents\\10minweather\\HENGCHUN10min.csv',encoding='utf-8-sig')