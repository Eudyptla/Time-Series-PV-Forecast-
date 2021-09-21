# importing modules
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
from datetime import datetime

# Collect data
# Analysis CSS
url = "https://www.cwb.gov.tw/V8/E/W/Observe/MOD/24hr/46759.html?T=698"
html = urlopen(url).read()
soup = BeautifulSoup(html, "lxml")
# check time
time_log = soup.find_all('th')
data_time = [i.get_text() for i in time_log]
# Add year
now_year = str(datetime.now().year) + '/'
data_time_year = [now_year + i for i in data_time]
data = pd.DataFrame({'time': data_time_year})
# other data
soup_td = soup.find_all('td')
headers_set = set([i['headers'][0] for i in soup_td])
headers_list = list(headers_set)
headers_list.sort()
# add data
for i in headers_list[:-1]:
    data_value = soup.find_all('td', headers=i)
    data[i] = [j.get_text() for j in data_value]

# weather data different
weather_data = soup.find_all('td', headers='weather')
data['weather'] = [i.img['alt'] for i in weather_data]
# save data
data.to_csv('data\\CWB_Heng_Chun .csv', encoding='utf-8-sig')


