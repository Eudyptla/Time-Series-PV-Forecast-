import pandas as pd

data_o =pd.DataFrame([])
for Year in range(2016,2018):
    for Month in range(1,13):
        for j in range(1,4):
            if j==1:
                date = '上旬'
                day_start = 1
                day_end = 11

            elif j==2:
                date = '中旬'
                day_start = 11
                day_end = 21

            else:
                date = '下旬'
                day_start = 21
                day_end = 1

            if Month>9:
                xls = pd.ExcelFile(
                    'E:\\Users\\Documents\\Taichu_DE\\' + str(Year) + '年' + str(Month) + '月' + date + '-(S04).xlsx')
            else:
                xls = pd.ExcelFile(
                    'E:\\Users\\Documents\\Taichu_DE\\' + str(Year) + '年0' + str(Month) + '月' + date + '-(S04).xlsx')

            df_1 = pd.read_excel(xls,'5.自存檔案的格試')
            if j == 3:
                if Month == 2:
                    if Year == 2016:
                        hour = df_1.iloc[4:22, 13:25]
                        hour = hour.stack(dropna=False)
                    else:
                        hour = df_1.iloc[4:20, 13:25]
                        hour = hour.stack(dropna=False)

                elif (Month == 4 or Month == 6 or Month == 9 or Month == 11):
                    hour = df_1.iloc[4:24, 13:25]
                    hour = hour.stack(dropna=False)
                else:
                    hour = df_1.iloc[4:26, 13:25]
                    hour = hour.stack(dropna=False)

            else:

                hour = df_1.iloc[4:24, 13:25]
                hour = hour.stack(dropna=False)

            date_start = pd.Timestamp(Year, Month, day_start, 1)
            if j == 3:
                if Month == 12:
                    date_end = pd.Timestamp(Year + 1, 1, day_end)
                else:
                    date_end = pd.Timestamp(Year, Month + 1, day_end)
            else:
                date_end = pd.Timestamp(Year, Month, day_end)



            time_Hour = pd.date_range(start = date_start,end = date_end,freq='H')
            data_F = pd.DataFrame(hour.values, index=time_Hour,columns=['power(KWH)'])
            data_o = data_o.append(data_F)
            data_o.to_csv('Taichung_DE.csv', encoding='utf-8')