# -*- coding: utf-8 -*-
#import tensorflow as tf
import pandas as pd
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from collections import defaultdict
import itertools
from sklearn.preprocessing import MinMaxScaler

# term structure
def main():
    
    #term structures
    xls = pd.ExcelFile('./project_VIX_future_contracts.xlsx')
    months = ['19 Sep','19 Oct','19 Nov','19 Dec','20 Jan','20 Feb','20 Mar','20 Apr','20 May','20 Jun','20 Jul','20 Aug','20 Sep','20 Oct','20 Nov','20 Dec','21 Jan'] 
    dates = ['2019-08-19','2019-09-18',
        '2019-09-19','2019-10-16',
        '2019-10-17','2019-11-20',
        '2019-11-21','2019-12-18',
        '2019-12-19','2020-01-22',
        '2020-01-23','2020-02-19',
        '2020-02-20','2020-03-18',
        '2020-03-19','2020-04-15',
        '2020-04-16','2020-05-20',
        '2020-05-21','2020-06-17']
    pairs = list(itertools.combinations([8,7,6,5,4,3,2], 2))
    
    for j in range(10):
        final_date = pd.bdate_range(dates[2*j], dates[2*j+1]).tolist()
        for i in range(j,j+8):
            df = pd.read_excel(xls, months[i])
            df = df.rename(columns={'Unnamed: 0':"index", 'Unnamed: 1':"date_open", 'Unnamed: 3':"date_last", 'Unnamed: 5':"date_high", 'Unnamed: 7':"date_low"})

            df_open = df[df['date_open'].isin(final_date)][['date_open']].reset_index(drop=True)
            df_last = df[df['date_last'].isin(final_date)][['PX_LAST']].reset_index(drop=True)
            df_high = df[df['date_high'].isin(final_date)][['PX_HIGH']].reset_index(drop=True)
            df_low = df[df['date_low'].isin(final_date)][['PX_LOW']].reset_index(drop=True)
            df_concat = pd.concat([df_open, df_last, df_high, df_low], axis=1, ignore_index=True)
            df_concat.columns = ['date', 'm_{}_close'.format(i-j+1), 'm_{}_high'.format(i-j+1), 'm_{}_low'.format(i-j+1)]

            df_concat['m_{}_hl'.format(i-j+1)] = df_concat.apply(lambda x: x['m_{}_high'.format(i-j+1)] - x['m_{}_low'.format(i-j+1)], axis=1)
            df_concat = df_concat.drop(columns=['m_{}_high'.format(i-j+1), 'm_{}_low'.format(i-j+1)])

            if i == j:
                df_final = df_concat
            else:
                df_final = pd.merge(df_final, df_concat, on='date') 
        
        for pair in pairs:
            df_final['m_{}-{}'.format(pair[0],pair[1])] = df_final['m_{}_close'.format(pair[0])] - df_final['m_{}_close'.format(pair[1])]

        for i in range(1,9):
            df_final = df_final.drop(columns=['m_{}_close'.format(i)])

        if j== 0:
            df_final_combine = df_final
        else:
            df_final_combine = df_final_combine.append(df_final)
    
    df_final_combine = df_final_combine.drop(columns=['m_1_hl'])
    
    #print(df_final_combine.head(50))
    

    #technical variables
    xls = pd.ExcelFile('./project_VIX_Related.xlsm')
    df_vix = pd.read_excel(xls)[['date','vix']]
    date_range = pd.bdate_range('2019-01-19', '2020-06-19').tolist()
    df_vix_range = df_vix[df_vix['date'].isin(date_range)].reset_index(drop=True)

    def prior_avg(datafarme, date, day):
        idx = datafarme[datafarme['date'] == date].index[0]
        sum_ = 0
        for i in range(idx-day,idx-1):
            sum_ += datafarme.loc[i]['vix']
        avg = sum_/day
        return avg
    
    def avg_above(datafarme, date, day):
        avg = prior_avg(datafarme, date, day)
        spot = datafarme[datafarme['date'] == date]['vix'].values[0]
        result = spot - avg > 0
        return 1 if result else 0

    def avg_below(datafarme, date, day):
        avg = prior_avg(datafarme, date, day)
        spot = datafarme[datafarme['date'] == date]['vix'].values[0]
        result = spot - avg < 0
        return 1 if result else 0
    
    def three_day_above(datafarme, date, day):
        idx = datafarme.index[datafarme['date']==date].tolist()[0]
        d1 = avg_above(datafarme, datafarme.iloc[idx-1]['date'], day)
        d2 = avg_above(datafarme, datafarme.iloc[idx-2]['date'], day)
        d3 = avg_above(datafarme, datafarme.iloc[idx-3]['date'], day)
        if d1 == d2 == d3 == 1:
            return 1
        else:
            return 0

    def bollinger_bands(datafarme, date, type_):
        spot = datafarme[datafarme['date'] == date]['vix'].values[0]
        avg = prior_avg(datafarme, date, 21)
        idx = datafarme.index[datafarme['date']==date].tolist()[0]
        std = datafarme.iloc[idx-21:idx]['vix'].std()
        if type_ == 'above':
            return 1 if spot > avg+2*std else 0
        elif type_ == 'below':
            return 1 if spot < avg-2*std else 0

    p_14_above, p_50_above, p_100_above = [], [], []
    p_14_below, p_50_below, p_100_below = [], [], []
    p_14_3day_above, p_50_3day_above, p_100_3day_above = [], [], []
    bollinger_bands_above, bollinger_bands_below = [], []
    for date in df_final_combine['date'].tolist():
        p_14_above.append(avg_above(df_vix_range, date, 14))
        p_50_above.append(avg_above(df_vix_range, date, 50))
        p_100_above.append(avg_above(df_vix_range, date, 100))
        p_14_below.append(avg_below(df_vix_range, date, 14))
        p_50_below.append(avg_below(df_vix_range, date, 50))
        p_100_below.append(avg_below(df_vix_range, date, 100))
        p_14_3day_above.append(three_day_above(df_vix_range, date, 14))
        p_50_3day_above.append(three_day_above(df_vix_range, date, 50))
        p_100_3day_above.append(three_day_above(df_vix_range, date, 100))
        bollinger_bands_above.append(bollinger_bands(df_vix_range, date, 'above'))
        bollinger_bands_below.append(bollinger_bands(df_vix_range, date, 'below'))

    df_final_combine['p_14_above'] = p_14_above
    df_final_combine['p_50_above'] = p_50_above
    df_final_combine['p_100_above'] = p_100_above
    df_final_combine['p_14_below'] = p_14_below
    df_final_combine['p_50_below'] = p_50_below
    df_final_combine['p_100_below'] = p_100_below
    df_final_combine['p_14_3day_above'] = p_14_3day_above
    df_final_combine['p_50_3day_above'] = p_50_3day_above
    df_final_combine['p_100_3day_above'] = p_100_3day_above
    df_final_combine['BB_above'] = bollinger_bands_above
    df_final_combine['BB_below'] = bollinger_bands_below
    
    # print(df_final_combine.head(50))


    ##vvix
    xls = pd.ExcelFile('./VIX_VVIX_new.xlsx')
    df_vvix = pd.read_excel(xls)
    df_vvix['vvix_hl'] = df_vvix.apply(lambda x: x['high price'] - x['low price'], axis=1)
    df_vvix = df_vvix.drop(columns=['VIX Index date', 'VIX Index low price', 'VIX Index high price', 'VIX Index last price'])
    # print(df_vvix.head(10))    
    vvix = []
    vvix_hl = []
    for date in df_final_combine['date'].tolist():
        try:
            vvix.append(df_vvix[df_vvix['VVIX Index date'] == date]['last price'].values[0])
            vvix_hl.append(df_vvix[df_vvix['VVIX Index date'] == date]['vvix_hl'].values[0])
        except:
            print('failed date: {}'.format(date))
    df_final_combine['vvix'] = vvix
    df_final_combine['vvix_hl'] = vvix_hl

    # print(df_final_combine.head(50))


    ##ground_truth
    xls = pd.ExcelFile('./project_VIX_future_contracts.xlsx')
    months = ['19 Sep','19 Oct','19 Nov','19 Dec','20 Jan','20 Feb','20 Mar','20 Apr','20 May','20 Jun','20 Jul','20 Aug','20 Sep','20 Oct','20 Nov','20 Dec','21 Jan'] 
    dates = ['2019-08-19','2019-09-18',
        '2019-09-19','2019-10-16',
        '2019-10-17','2019-11-20',
        '2019-11-21','2019-12-18',
        '2019-12-19','2020-01-22',
        '2020-01-23','2020-02-19',
        '2020-02-20','2020-03-18',
        '2020-03-19','2020-04-15',
        '2020-04-16','2020-05-20',
        '2020-05-21','2020-06-17',
        '2020-06-18','2020-07-22']
    
    for i in range(11):
        final_date = pd.bdate_range(dates[2*i], dates[2*i+1]).tolist()

        df = pd.read_excel(xls, months[i])
        df = df.rename(columns={'Unnamed: 0':"index", 'Unnamed: 1':"date_open", 'Unnamed: 3':"date_last", 'Unnamed: 5':"date_high", 'Unnamed: 7':"date_low"})

        df_open = df[df['date_open'].isin(final_date)][['date_open']].reset_index(drop=True)
        df_last = df[df['date_last'].isin(final_date)][['PX_LAST']].reset_index(drop=True)
        df_concat = pd.concat([df_open, df_last], axis=1, ignore_index=True)
        df_concat.columns = ['date', 'm_1_close']
        if i == 0:
            date_list = df_concat['date'].to_list()
            m1_list = df_concat['m_1_close'].to_list()
        else:
            date_list += df_concat['date'].to_list()
            m1_list += df_concat['m_1_close'].to_list()
    
    gt_3 = m1_list[3:len(df_final_combine)+3]
    date_3 = date_list[3:len(df_final_combine)+3]
    gt_5 = m1_list[5:len(df_final_combine)+5]
    date_5 = date_list[5:len(df_final_combine)+5]

    # print(date_3[0], gt_3[0], date_3[-1], gt_3[-1])
    # print(date_5[0], date_5[-1])

    df_final_combine['gt_3'] = gt_3
    df_final_combine['gt_5'] = gt_5

    print(df_final_combine.head(50))

    
    ##output
    df_final_combine.to_pickle('./vix_future_preprocessed.pkl')
    print('completed')


if __name__ == "__main__":
    # main()
    import os
    from configs import get_config
    config = get_config()
    print(os.path.isfile(config.ckpt_path))
