from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import numpy as np
import time

def append_new_data(ori_data, url, append_date):
    time.sleep(0.5)
    url_date = url + str(append_date)[:11]
    end_date = pd.to_datetime([1], unit='D', origin=pd.Timestamp(append_date))
    time_hour = pd.date_range(start=append_date, end=end_date[0], freq='H')[1:]
    html = urlopen(url_date).read()
    soup = BeautifulSoup(html, "lxml")
    # get columns text
    c_tag = soup.find_all(class_='second_tr')
    c_text = [i.get_text().split('\n') for i in c_tag][0]
    c_text = [i.replace('\t', '') for i in c_text if i != '']
    # get data value
    v_tag = soup.find_all('td')
    v_value = [i.get_text().replace('\xa0', '') for i in v_tag]
    data_value = np.array(v_value[10:]).reshape(24, -1)
    append_df = pd.DataFrame(data_value, columns=c_text, index=time_hour)
    new_df = pd.concat([ori_data, append_df])
    return new_df

url = "http://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=467770&stname=%25E6%25A2%25A7%25E6%25A3%25B2&datepicker="
start_date = '2016-01-01'
final_date = '2017-12-31'
Location = 'WUQI '
my_df = pd.DataFrame()
time_delta = pd.date_range(start=start_date, end=final_date, freq='D')
for date_i in time_delta:
    my_df = append_new_data(my_df, url, date_i)
    print(date_i, 'is finished')

my_df.to_csv('data\\'+Location+'.csv', encoding='utf-8-sig')