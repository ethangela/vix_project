# -*- coding: utf-8 -*-
import pandas as pd
from collections import defaultdict
import itertools
from sklearn.preprocessing import MinMaxScaler
import os
import math
import numpy as np

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




def main_new():
    # ## new vix project ##     
    
    # # term structure
    # mth_code = {'VXF':1, 'VXG':2, 'VXH':3, 'VXJ':4, 'VXK':5, 'VXM':6, 'VXN':7, 'VXQ':8, 'VXU':9, 'VXV':10, 'VXX':11, 'VXZ':12}
    # pairs = list(itertools.combinations([8,7,6,5,4,3,2,1], 2))
    # c = 0
    # for year in ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']:
    #     subdirs = [ f.path for f in os.scandir(year+'/') if f.is_dir() ] #2011/20110101, 2011/20110102, ...
    #     subdirs = sorted(subdirs)
    #     b_dates = pd.bdate_range(year+'/01/01', year+'/12/31')
    #     for i,subdir in enumerate(subdirs): #0, 2011/20110101 ...
    #         date = subdir.split('/')[-1]
    #         if date in b_dates:
    #             df_dic = {}
    #             df_dic['date'] = date
    #             futures = sorted(os.listdir(subdir)) #VXF1, VXG!, ... 
    #             curt_mth = int(subdir.split('/')[-1][4:6]) #1
    #             # print(date, curt_mth, mth_code[futures[0][:3]])
    #             if curt_mth == 1 and mth_code[futures[0][:3]] == 1:
    #                 for future in futures: #default 8
    #                         fut_mth = mth_code[future[:3]] 
    #                         if fut_mth > curt_mth:
    #                             trm_mth = fut_mth - curt_mth + 1 
    #                         elif fut_mth == curt_mth:
    #                             trm_mth = 1
    #                         if trm_mth >= 9:
    #                             continue
    #                         df = pd.read_csv(os.path.join(subdir, future), compression='gzip')
    #                         intraday_high = df['HighTradePrice'].max()
    #                         intraday_low = df['LowTradePrice'].min()
    #                         try:
    #                             settle_close = [x for x in df['CloseTradePrice'].tolist() if (math.isnan(x) == False)][-1]
    #                         except:
    #                             settle_close = math.nan   
    #                         df_dic['m_{}_close'.format(trm_mth)] = settle_close
    #                         df_dic['m_{}_hl'.format(trm_mth)] = intraday_high - intraday_low  
    #                         df_dic['m_{}_high'.format(trm_mth)] = intraday_high
    #                         df_dic['m_{}_low'.format(trm_mth)] = intraday_low
    #             elif curt_mth < 5 and mth_code[futures[0][:3]] != 1:
    #                 if mth_code[futures[0][:3]] == curt_mth:
    #                     for future in futures: #default 8
    #                         fut_mth = mth_code[future[:3]] 
    #                         if fut_mth > curt_mth:
    #                             trm_mth = fut_mth - curt_mth + 1 
    #                         elif fut_mth == curt_mth:
    #                             trm_mth = 1
    #                         else:
    #                             trm_mth = fut_mth + 13 - curt_mth
    #                         if trm_mth >= 9:
    #                             continue
    #                         df = pd.read_csv(os.path.join(subdir, future), compression='gzip')
    #                         intraday_high = df['HighTradePrice'].max()
    #                         intraday_low = df['LowTradePrice'].min()
    #                         try:
    #                             settle_close = [x for x in df['CloseTradePrice'].tolist() if (math.isnan(x) == False)][-1]
    #                         except:
    #                             settle_close = math.nan   
    #                         df_dic['m_{}_close'.format(trm_mth)] = settle_close
    #                         df_dic['m_{}_hl'.format(trm_mth)] = intraday_high - intraday_low  
    #                         df_dic['m_{}_high'.format(trm_mth)] = intraday_high
    #                         df_dic['m_{}_low'.format(trm_mth)] = intraday_low
    #                 else:
    #                     for future in futures:
    #                         fut_mth = mth_code[future[:3]] 
    #                         if fut_mth > curt_mth:
    #                             trm_mth = fut_mth - curt_mth
    #                         else:
    #                             trm_mth = fut_mth + 13 - curt_mth
    #                         if trm_mth >= 9:
    #                             continue
    #                         df = pd.read_csv(os.path.join(subdir, future), compression='gzip')
    #                         intraday_high = df['HighTradePrice'].max()
    #                         intraday_low = df['LowTradePrice'].min()
    #                         try:
    #                             settle_close = [x for x in df['CloseTradePrice'].tolist() if (math.isnan(x) == False)][-1]
    #                         except:
    #                             settle_close = math.nan     
    #                         df_dic['m_{}_close'.format(trm_mth)] = settle_close
    #                         df_dic['m_{}_hl'.format(trm_mth)] = intraday_high - intraday_low  
    #                         df_dic['m_{}_high'.format(trm_mth)] = intraday_high
    #                         df_dic['m_{}_low'.format(trm_mth)] = intraday_low
    #             else:
    #                 if curt_mth in [mth_code[x[:3]] for x in futures]:
    #                     for future in futures:
    #                         fut_mth = mth_code[future[:3]] 
    #                         if fut_mth < curt_mth:
    #                             trm_mth = fut_mth + 13 - curt_mth
    #                         else:
    #                             trm_mth = fut_mth + 1 - curt_mth
    #                         if trm_mth >= 9:
    #                             continue
    #                         df = pd.read_csv(os.path.join(subdir, future), compression='gzip')
    #                         intraday_high = df['HighTradePrice'].max()
    #                         intraday_low = df['LowTradePrice'].min()
    #                         try:
    #                             settle_close = [x for x in df['CloseTradePrice'].tolist() if (math.isnan(x) == False)][-1]
    #                         except:
    #                             settle_close = math.nan   
    #                         df_dic['m_{}_close'.format(trm_mth)] = settle_close
    #                         df_dic['m_{}_hl'.format(trm_mth)] = intraday_high - intraday_low  
    #                         df_dic['m_{}_high'.format(trm_mth)] = intraday_high
    #                         df_dic['m_{}_low'.format(trm_mth)] = intraday_low
    #                 else:
    #                     for future in futures:
    #                         fut_mth = mth_code[future[:3]] 
    #                         if fut_mth < curt_mth:
    #                             trm_mth = fut_mth + 12 - curt_mth
    #                         else:
    #                             trm_mth = fut_mth - curt_mth
    #                         if trm_mth >= 9:
    #                             continue
    #                         df = pd.read_csv(os.path.join(subdir, future), compression='gzip')
    #                         intraday_high = df['HighTradePrice'].max()
    #                         intraday_low = df['LowTradePrice'].min()
    #                         try:
    #                             settle_close = [x for x in df['CloseTradePrice'].tolist() if (math.isnan(x) == False)][-1]
    #                         except:
    #                             settle_close = math.nan   
    #                         df_dic['m_{}_close'.format(trm_mth)] = settle_close
    #                         df_dic['m_{}_hl'.format(trm_mth)] = intraday_high - intraday_low  
    #                         df_dic['m_{}_high'.format(trm_mth)] = intraday_high
    #                         df_dic['m_{}_low'.format(trm_mth)] = intraday_low   
                
    #             if c == 0:
    #                 df_final = pd.DataFrame([df_dic])
    #             else:
    #                 df_final = df_final.append(df_dic, ignore_index=True)               
    #             c += 1
    #         else:
    #             continue
    # for pair in pairs:
    #     df_final.loc[:,'m_{}-{}'.format(pair[0],pair[1])] = df_final['m_{}_close'.format(pair[0])] - df_final['m_{}_close'.format(pair[1])] 
    # for i in range(2,9):
    #     df_final = df_final.drop(columns=['m_{}_close'.format(i)])
    #     df_final = df_final.drop(columns=['m_{}_high'.format(i)])
    #     df_final = df_final.drop(columns=['m_{}_low'.format(i)])
    # df_final = df_final.fillna(method='ffill')
    # print(df_final)
    # df_final.to_pickle('./vix_future_tmp_1.pkl')
    


    # # #technical variables for 1mth vix future AND vix OR/AND vvix
    # df_final = pd.read_pickle('./vix_future_tmp_1.pkl')
    # xls = pd.ExcelFile('./VIX_2011-2021.xlsx')
    # df_vix = pd.read_excel(xls, header=0)
    # invalid_date_due_to_vix = []
    # for date in df_final['date'].tolist():
    #     if len(df_vix[df_vix['date'] == date]['close'].values) != 0:
    #         continue
    #     else:
    #         invalid_date_due_to_vix.append(date)
    #         df_final = df_final.drop(df_final[df_final['date'] == date].index)
    # df_final = df_final.reset_index(drop=True)
    # print('invalid_date_due_to_vix: {}'.format(invalid_date_due_to_vix))

    
    # def prior_avg(datafarme, column, date, day):
    #     idx = datafarme[datafarme['date'] == date].index[0]
    #     sum_ = 0
    #     for i in range(idx-day+1,idx+1):
    #         sum_ += datafarme.loc[i][column]
    #     avg = sum_/day
    #     return avg
    
    # def avg_move(datafarme, column, date, day, type_):
    #     avg = prior_avg(datafarme, column, date, day)
    #     spot = datafarme[datafarme['date'] == date][column].values[0]
    #     if type_ == 'above':
    #         result = spot - avg > 0
    #         return 1 if result else 0
    #     elif type_ == 'below':
    #         result = spot - avg < 0
    #         return 1 if result else 0
            
    # def avg_three_day_move(datafarme, column, date, day, type_):
    #     idx = datafarme.index[datafarme['date']==date].tolist()[0]
    #     if type_ == 'above':
    #         d1 = avg_move(datafarme, column, datafarme.iloc[idx-1]['date'], day, type_)
    #         d2 = avg_move(datafarme, column, datafarme.iloc[idx-2]['date'], day, type_)
    #         d3 = avg_move(datafarme, column, datafarme.iloc[idx-3]['date'], day, type_)
    #         if d1 == d2 == d3 == 1:
    #             return 1
    #         else:
    #             return 0
    #     elif type_ == 'below':
    #         d1 = avg_move(datafarme, column, datafarme.iloc[idx-1]['date'], day, type_)
    #         d2 = avg_move(datafarme, column, datafarme.iloc[idx-2]['date'], day, type_)
    #         d3 = avg_move(datafarme, column, datafarme.iloc[idx-3]['date'], day, type_)
    #         if d1 == d2 == d3 == 1:
    #             return 1
    #         else:
    #             return 0

    # def bollinger_bands_move(datafarme, column, date, type_):
    #     spot = datafarme[datafarme['date'] == date][column].values[0]
    #     avg = prior_avg(datafarme, column, date, 21)
    #     idx = datafarme.index[datafarme['date']==date].tolist()[0]
    #     std = datafarme.iloc[idx-21:idx][column].std()
    #     if type_ == 'above':
    #         return 1 if spot > avg+2*std else 0
    #     elif type_ == 'below':
    #         return 1 if spot < avg-2*std else 0

    # def exponential_avg(datafarme, column, date, day):
    #     idx = datafarme.index[datafarme['date']==date].tolist()[0]
    #     avg_t_1 = prior_avg(datafarme, column, datafarme.iloc[idx-1]['date'], day)
    #     spot = datafarme[datafarme['date'] == date][column].values[0]
    #     exp = (spot - avg_t_1) * 2 / (day+1) + avg_t_1
    #     return exp

    # def exponential_avg_move(datafarme, column, date, day, type_):
    #     exp_avg = exponential_avg(datafarme, column, date, day)
    #     spot = datafarme[datafarme['date'] == date][column].values[0]
    #     if type_ == 'above':
    #         result = spot - exp_avg > 0
    #         return 1 if result else 0
    #     elif type_ == 'below':
    #         result = spot - exp_avg < 0
    #         return 1 if result else 0

    # def disparity(datafarme, column, date, day):
    #     avg = prior_avg(datafarme, column, date, day)
    #     spot = datafarme[datafarme['date'] == date][column].values[0]
    #     return spot / avg * 100
    
    # def disparity_move(datafarme, column, date, day, type_):
    #     disp_today = disparity(datafarme, column, date, day)
    #     idx = datafarme.index[datafarme['date']==date].tolist()[0]
    #     disp_yesterday = disparity(datafarme, column, datafarme.iloc[idx-1]['date'], day)
    #     if type_ == 'above':
    #         result = disp_today - disp_yesterday > 0
    #         return 1 if result else 0
    #     elif type_ == 'below':
    #         result = disp_today - disp_yesterday < 0
    #         return 1 if result else 0

    # def momentum1(datafarme, column, date):
    #     spot = datafarme[datafarme['date'] == date][column].values[0]
    #     idx = datafarme.index[datafarme['date']==date].tolist()[0]
    #     prior = datafarme.iloc[idx-5][column]#.values[0]
    #     return spot / prior * 100

    # def momentum2(datafarme, column, date):
    #     spot = datafarme[datafarme['date'] == date][column].values[0]
    #     idx = datafarme.index[datafarme['date']==date].tolist()[0]
    #     prior = datafarme.iloc[idx-5][column]#.values[0]
    #     return (spot - prior + 0.001) * 100

    # def ROC(datafarme, column, date):
    #     spot = datafarme[datafarme['date'] == date][column].values[0]
    #     mtm = momentum2(datafarme, column, date)
    #     return spot / mtm * 100

    # def ROC_move(datafarme, column, date, type_):
    #     roc_today = ROC(datafarme, column, date)
    #     idx = datafarme.index[datafarme['date']==date].tolist()[0]
    #     roc_yesterday = ROC(datafarme, column, datafarme.iloc[idx-1]['date'])
    #     if type_ == 'above':
    #         result = roc_today - roc_yesterday > 0
    #         return 1 if result else 0
    #     elif type_ == 'below':
    #         result = roc_today - roc_yesterday < 0
    #         return 1 if result else 0

    # def stochastic_william(datafarme, column, date, day=14, type_='william'):
    #     idx = datafarme[datafarme['date'] == date].index[0]
    #     spot = datafarme[datafarme['date'] == date][column+'close'].values[0]
    #     highest_high = -99
    #     lowest_low = 99
    #     for i in range(idx-day+1,idx+1):
    #         if datafarme.loc[i][column+'high'] > highest_high:
    #             highest_high = datafarme.loc[i][column+'high']
    #         if datafarme.loc[i][column+'low'] < lowest_low:
    #             lowest_low = datafarme.loc[i][column+'low']
    #     if type_ == 'william':
    #         return (highest_high-spot) / (highest_high-lowest_low)
    #     elif type_ == 'stochastic':
    #         return (spot-lowest_low) / (highest_high-lowest_low) * 100
            

    # vix_p_3_avg, vix_p_5_avg, vix_p_10_avg = [], [], []
    # vix_p_3_above, vix_p_5_above, vix_p_10_above = [], [], []
    # vix_p_3_below, vix_p_5_below, vix_p_10_below = [], [], []
    # vix_p_3_above_3d, vix_p_5_above_3d, vix_p_10_above_3d = [], [], []
    # vix_p_3_below_3d, vix_p_5_below_3d, vix_p_10_below_3d = [], [], []
    # vix_bollinger_bands_above, vix_bollinger_bands_below = [], []
    # vix_p_3_exp_avg, vix_p_5_exp_avg, vix_p_10_exp_avg = [], [], []
    # vix_p_3_exp_above, vix_p_5_exp_above, vix_p_10_exp_above = [], [], []
    # vix_p_3_exp_below, vix_p_5_exp_below, vix_p_10_exp_below = [], [], []
    # vix_disp_3, vix_disp_5 = [], []
    # vix_disp_3_above, vix_disp_5_above = [], []
    # vix_disp_3_below, vix_disp_5_below = [], []
    # vix_mom1_5, vix_mom2_5 = [], []
    # vix_roc, vix_roc_above, vix_roc_below = [], [], []
    # vix_william = []
    # vix_stochastic = []


    # fut_p_3_avg, fut_p_5_avg, fut_p_10_avg = [], [], []
    # fut_p_3_above, fut_p_5_above, fut_p_10_above = [], [], []
    # fut_p_3_below, fut_p_5_below, fut_p_10_below = [], [], []
    # fut_p_3_above_3d, fut_p_5_above_3d, fut_p_10_above_3d = [], [], []
    # fut_p_3_below_3d, fut_p_5_below_3d, fut_p_10_below_3d = [], [], []
    # fut_bollinger_bands_above, fut_bollinger_bands_below = [], []
    # fut_p_3_exp_avg, fut_p_5_exp_avg, fut_p_10_exp_avg = [], [], []
    # fut_p_3_exp_above, fut_p_5_exp_above, fut_p_10_exp_above = [], [], []
    # fut_p_3_exp_below, fut_p_5_exp_below, fut_p_10_exp_below = [], [], []
    # fut_disp_3, fut_disp_5 = [], []
    # fut_disp_3_above, fut_disp_5_above = [], []
    # fut_disp_3_below, fut_disp_5_below = [], []
    # fut_mom1_5, fut_mom2_5 = [], []
    # fut_roc, fut_roc_above, fut_roc_below = [], [], []
    # fut_william = []
    # fut_stochastic = []


    # for date in df_final['date'].tolist()[20:]: 
    #     #moving average
    #     vix_p_3_avg.append(prior_avg(df_vix, 'close', date, 3))
    #     vix_p_5_avg.append(prior_avg(df_vix, 'close', date, 5))
    #     vix_p_10_avg.append(prior_avg(df_vix, 'close', date, 10))

    #     fut_p_3_avg.append(prior_avg(df_final, 'm_1_close', date, 3))
    #     fut_p_5_avg.append(prior_avg(df_final, 'm_1_close', date, 5))
    #     fut_p_10_avg.append(prior_avg(df_final, 'm_1_close', date, 10))
        
    #     #moving average move
    #     vix_p_3_above.append(avg_move(df_vix, 'close', date, 3, 'above'))
    #     vix_p_5_above.append(avg_move(df_vix, 'close', date, 5, 'above'))
    #     vix_p_10_above.append(avg_move(df_vix, 'close', date, 10, 'above'))
    #     vix_p_3_below.append(avg_move(df_vix, 'close', date, 3, 'below'))
    #     vix_p_5_below.append(avg_move(df_vix, 'close', date, 5, 'below'))
    #     vix_p_10_below.append(avg_move(df_vix, 'close', date, 10, 'below'))
    #     vix_p_3_above_3d.append(avg_three_day_move(df_vix, 'close', date, 3, 'above'))
    #     vix_p_5_above_3d.append(avg_three_day_move(df_vix, 'close', date, 5, 'above'))
    #     vix_p_10_above_3d.append(avg_three_day_move(df_vix, 'close', date, 10, 'above'))
    #     vix_p_3_below_3d.append(avg_three_day_move(df_vix, 'close', date, 3, 'below'))
    #     vix_p_5_below_3d.append(avg_three_day_move(df_vix, 'close', date, 5, 'below'))
    #     vix_p_10_below_3d.append(avg_three_day_move(df_vix, 'close', date, 10, 'below'))

    #     fut_p_3_above.append(avg_move(df_final, 'm_1_close', date, 3, 'above'))
    #     fut_p_5_above.append(avg_move(df_final, 'm_1_close', date, 5, 'above'))
    #     fut_p_10_above.append(avg_move(df_final, 'm_1_close', date, 10, 'above'))
    #     fut_p_3_below.append(avg_move(df_final, 'm_1_close', date, 3, 'below'))
    #     fut_p_5_below.append(avg_move(df_final, 'm_1_close', date, 5, 'below'))
    #     fut_p_10_below.append(avg_move(df_final, 'm_1_close', date, 10, 'below'))
    #     fut_p_3_above_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 3, 'above'))
    #     fut_p_5_above_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 5, 'above'))
    #     fut_p_10_above_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 10, 'above'))
    #     fut_p_3_below_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 3, 'below'))
    #     fut_p_5_below_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 5, 'below'))
    #     fut_p_10_below_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 10, 'below'))


    #     #bollinger band move
    #     vix_bollinger_bands_above.append(bollinger_bands_move(df_vix, 'close', date, 'above'))
    #     vix_bollinger_bands_below.append(bollinger_bands_move(df_vix, 'close', date, 'below'))

    #     fut_bollinger_bands_above.append(bollinger_bands_move(df_final, 'm_1_close', date, 'above'))
    #     fut_bollinger_bands_below.append(bollinger_bands_move(df_final, 'm_1_close', date, 'below'))


    #     #exponential moving average
    #     vix_p_3_exp_avg.append(exponential_avg(df_vix, 'close', date, 3))
    #     vix_p_5_exp_avg.append(exponential_avg(df_vix, 'close', date, 5))
    #     vix_p_10_exp_avg.append(exponential_avg(df_vix, 'close', date, 10))
        
    #     fut_p_3_exp_avg.append(exponential_avg(df_final, 'm_1_close', date, 3))
    #     fut_p_5_exp_avg.append(exponential_avg(df_final, 'm_1_close', date, 5))
    #     fut_p_10_exp_avg.append(exponential_avg(df_final, 'm_1_close', date, 10))
        

    #     #exponential moving average move
    #     vix_p_3_exp_above.append(exponential_avg_move(df_vix, 'close', date, 3, 'above'))
    #     vix_p_5_exp_above.append(exponential_avg_move(df_vix, 'close', date, 5, 'above'))
    #     vix_p_10_exp_above.append(exponential_avg_move(df_vix, 'close', date, 10, 'above'))
    #     vix_p_3_exp_below.append(exponential_avg_move(df_vix, 'close', date, 3, 'below'))
    #     vix_p_5_exp_below.append(exponential_avg_move(df_vix, 'close', date, 5, 'below'))
    #     vix_p_10_exp_below.append(exponential_avg_move(df_vix, 'close', date, 10, 'below'))
        
    #     fut_p_3_exp_above.append(exponential_avg_move(df_final, 'm_1_close', date, 3, 'above'))
    #     fut_p_5_exp_above.append(exponential_avg_move(df_final, 'm_1_close', date, 5, 'above'))
    #     fut_p_10_exp_above.append(exponential_avg_move(df_final, 'm_1_close', date, 10, 'above'))
    #     fut_p_3_exp_below.append(exponential_avg_move(df_final, 'm_1_close', date, 3, 'below'))
    #     fut_p_5_exp_below.append(exponential_avg_move(df_final, 'm_1_close', date, 5, 'below'))
    #     fut_p_10_exp_below.append(exponential_avg_move(df_final, 'm_1_close', date, 10, 'below'))
        
        
    #     #disparity
    #     vix_disp_3.append(disparity(df_vix, 'close', date, 3))
    #     vix_disp_5.append(disparity(df_vix, 'close', date, 5))
        
    #     fut_disp_3.append(disparity(df_final, 'm_1_close', date, 3))
    #     fut_disp_5.append(disparity(df_final, 'm_1_close', date, 5))


    #     #disparity move
    #     vix_disp_3_above.append(disparity_move(df_vix, 'close', date, 3, 'above'))
    #     vix_disp_5_above.append(disparity_move(df_vix, 'close', date, 5, 'above'))
    #     vix_disp_3_below.append(disparity_move(df_vix, 'close', date, 3, 'below'))
    #     vix_disp_5_below.append(disparity_move(df_vix, 'close', date, 5, 'below'))
        
    #     fut_disp_3_above.append(disparity_move(df_final, 'm_1_close', date, 3, 'above'))
    #     fut_disp_5_above.append(disparity_move(df_final, 'm_1_close', date, 5, 'above'))
    #     fut_disp_3_below.append(disparity_move(df_final, 'm_1_close', date, 3, 'below'))
    #     fut_disp_5_below.append(disparity_move(df_final, 'm_1_close', date, 5, 'below'))


    #     #momentum1
    #     vix_mom1_5.append(momentum1(df_vix, 'close', date))
    #     fut_mom1_5.append(momentum1(df_final, 'm_1_close', date))


    #     #momentum2
    #     vix_mom2_5.append(momentum2(df_vix, 'close', date))
    #     fut_mom2_5.append(momentum2(df_final, 'm_1_close', date))


    #     #ROC
    #     vix_roc.append(ROC(df_vix, 'close', date))
    #     fut_roc.append(ROC(df_final, 'm_1_close', date))
        

    #     #ROC move
    #     vix_roc_above.append(ROC_move(df_vix, 'close', date, 'above'))
    #     vix_roc_below.append(ROC_move(df_vix, 'close', date, 'below'))
        
    #     fut_roc_above.append(ROC_move(df_final, 'm_1_close', date, 'above'))
    #     fut_roc_below.append(ROC_move(df_final, 'm_1_close', date, 'below'))


    #     #william %R
    #     vix_william.append(stochastic_william(df_vix, '', date, day=14, type_='william'))
    #     fut_william.append(stochastic_william(df_final, 'm_1_', date, day=14, type_='william'))
        
    #     #stochastic oscillator
    #     vix_stochastic.append(stochastic_william(df_vix, '', date, day=14, type_='stochastic'))
    #     fut_stochastic.append(stochastic_william(df_final, 'm_1_', date, day=14, type_='stochastic'))
        

    # df_final = df_final.iloc[20:]
    # df_final = df_final.reset_index(drop=True)

    # df_final.loc[:,'vix_p_3_avg'] = vix_p_3_avg
    # df_final.loc[:,'vix_p_5_avg'] = vix_p_5_avg
    # df_final.loc[:,'vix_p_10_avg'] = vix_p_10_avg
    # df_final.loc[:,'vix_p_3_above'] = vix_p_3_above
    # df_final.loc[:,'vix_p_5_above'] = vix_p_5_above
    # df_final.loc[:,'vix_p_10_above'] = vix_p_10_above
    # df_final.loc[:,'vix_p_3_below'] = vix_p_3_below
    # df_final.loc[:,'vix_p_5_below'] = vix_p_5_below
    # df_final.loc[:,'vix_p_10_below'] = vix_p_10_below
    # df_final.loc[:,'fut_p_3_avg'] = fut_p_3_avg
    # df_final.loc[:,'fut_p_5_avg'] = fut_p_5_avg
    # df_final.loc[:,'fut_p_10_avg'] = fut_p_10_avg
    # df_final.loc[:,'fut_p_3_above'] = fut_p_3_above
    # df_final.loc[:,'fut_p_5_above'] = fut_p_5_above
    # df_final.loc[:,'fut_p_10_above'] = fut_p_10_above
    # df_final.loc[:,'fut_p_3_below'] = fut_p_3_below
    # df_final.loc[:,'fut_p_5_below'] = fut_p_5_below
    # df_final.loc[:,'fut_p_10_below'] = fut_p_10_below
    
    # df_final.loc[:,'vix_p_3_3day_above'] = vix_p_3_above_3d
    # df_final.loc[:,'vix_p_5_3day_above'] = vix_p_5_above_3d
    # df_final.loc[:,'vix_p_10_3day_above'] = vix_p_10_above_3d
    # df_final.loc[:,'vix_p_3_3day_below'] = vix_p_3_below_3d
    # df_final.loc[:,'vix_p_5_3day_below'] = vix_p_5_below_3d
    # df_final.loc[:,'vix_p_10_3day_below'] = vix_p_10_below_3d
    # df_final.loc[:,'fut_p_3_3day_above'] = fut_p_3_above_3d
    # df_final.loc[:,'fut_p_5_3day_above'] = fut_p_5_above_3d
    # df_final.loc[:,'fut_p_10_3day_above'] = fut_p_10_above_3d
    # df_final.loc[:,'fut_p_3_3day_below'] = fut_p_3_below_3d
    # df_final.loc[:,'fut_p_5_3day_below'] = fut_p_5_below_3d
    # df_final.loc[:,'fut_p_10_3day_below'] = fut_p_10_below_3d

    # df_final.loc[:,'vix_p_3_exp_avg'] = vix_p_3_exp_avg
    # df_final.loc[:,'vix_p_5_exp_avg'] = vix_p_5_exp_avg
    # df_final.loc[:,'vix_p_10_exp_avg'] = vix_p_10_exp_avg
    # df_final.loc[:,'vix_p_3_exp_above'] = vix_p_3_exp_above
    # df_final.loc[:,'vix_p_5_exp_above'] = vix_p_5_exp_above
    # df_final.loc[:,'vix_p_10_exp_above'] = vix_p_10_exp_above
    # df_final.loc[:,'vix_p_3_exp_below'] = vix_p_3_exp_below
    # df_final.loc[:,'vix_p_5_exp_below'] = vix_p_5_exp_below
    # df_final.loc[:,'vix_p_10_exp_below'] = vix_p_10_exp_below
    # df_final.loc[:,'fut_p_3_exp_avg'] = fut_p_3_exp_avg
    # df_final.loc[:,'fut_p_5_exp_avg'] = fut_p_5_exp_avg
    # df_final.loc[:,'fut_p_10_exp_avg'] = fut_p_10_exp_avg
    # df_final.loc[:,'fut_p_3_exp_above'] = fut_p_3_exp_above
    # df_final.loc[:,'fut_p_5_exp_above'] = fut_p_5_exp_above
    # df_final.loc[:,'fut_p_10_exp_above'] = fut_p_10_exp_above
    # df_final.loc[:,'fut_p_3_exp_below'] = fut_p_3_exp_below
    # df_final.loc[:,'fut_p_5_exp_below'] = fut_p_5_exp_below
    # df_final.loc[:,'fut_p_10_exp_below'] = fut_p_10_exp_below

    # df_final.loc[:,'vix_BB_above'] = vix_bollinger_bands_above
    # df_final.loc[:,'vix_BB_below'] = vix_bollinger_bands_below
    # df_final.loc[:,'fut_BB_above'] = fut_bollinger_bands_above
    # df_final.loc[:,'fut_BB_below'] = fut_bollinger_bands_below

    # df_final.loc[:,'vix_disp3_avg'] = vix_disp_3
    # df_final.loc[:,'vix_disp5_avg'] = vix_disp_5
    # df_final.loc[:,'vix_disp3_above'] = vix_disp_3_above
    # df_final.loc[:,'vix_disp5_above'] = vix_disp_5_above
    # df_final.loc[:,'vix_disp3_below'] = vix_disp_3_below
    # df_final.loc[:,'vix_disp5_below'] = vix_disp_5_below
    # df_final.loc[:,'fut_disp3_avg'] = fut_disp_3
    # df_final.loc[:,'fut_disp5_avg'] = fut_disp_5
    # df_final.loc[:,'fut_disp3_above'] = fut_disp_3_above
    # df_final.loc[:,'fut_disp5_above'] = fut_disp_5_above
    # df_final.loc[:,'fut_disp3_below'] = fut_disp_3_below
    # df_final.loc[:,'fut_disp5_below'] = fut_disp_5_below

    # df_final.loc[:,'vix_momentum1'] = vix_mom1_5
    # df_final.loc[:,'vix_momentum2'] = vix_mom2_5
    # df_final.loc[:,'vix_roc'] = vix_roc
    # df_final.loc[:,'vix_roc_above'] = vix_roc_above
    # df_final.loc[:,'vix_roc_below'] = vix_roc_below
    # df_final.loc[:,'fut_momentum1'] = fut_mom1_5
    # df_final.loc[:,'fut_momentum2'] = fut_mom2_5
    # df_final.loc[:,'fut_roc'] = fut_roc
    # df_final.loc[:,'fut_roc_above'] = fut_roc_above
    # df_final.loc[:,'fut_roc_below'] = fut_roc_below

    # df_final.loc[:,'vix_william'] = vix_william
    # df_final.loc[:,'vix_stochastic'] = vix_stochastic
    # df_final.loc[:,'fut_william'] = fut_william
    # df_final.loc[:,'fut_stochastic'] = fut_stochastic
    
    
    # print(df_final)
    # df_final.to_pickle('./vix_future_tmp_2.pkl')




    



    # ##vvix
    # df_final = pd.read_pickle('./vix_future_tmp_2.pkl')
    # xls = pd.ExcelFile('./VVIX_2012-2021.xlsx')
    # df_vvix = pd.read_excel(xls, header=0)
    # df_hl = df_vvix[['high date', 'VVIX high', 'VVIX low']]
    # df_close = df_vvix[['close date', 'VVIX close']]    
    # df_hl.loc[:,'vvix_hl'] = df_hl.apply(lambda x: x['VVIX high'] - x['VVIX low'], axis=1)
    # # print(df_vvix.head(10))    
    # vvix = []
    # vvix_hl = []
    # # invalid_vvix_date = []
    # for date in df_final['date'].tolist():
    #     if len(df_hl[df_hl['high date'] == date]['vvix_hl'].values) != 0 and len(df_close[df_close['close date'] == date]['VVIX close'].values) != 0:
    #         vvix_hl.append( df_hl[df_hl['high date'] == date]['vvix_hl'].values[0] )
    #         vvix.append( df_close[df_close['close date'] == date]['VVIX close'].values[0] )
    #     else:
    #         df_final = df_final.drop(df_final[df_final['date'] == date].index)
    #         # invalid_vvix_date.append(date)
    # df_final.loc[:,'vvix'] = vvix
    # df_final.loc[:,'vvix_hl'] = vvix_hl
    # df_final = df_final.reset_index(drop=True)
    # # print('invalid_vvix_date: {}'.format(invalid_vvix_date))

    
    # #rsi
    # xls = pd.ExcelFile('./RSI_data.xlsx')
    # df_rsi = pd.read_excel(xls)
    # # print(df_rsi)
    # d3, d9, d14, d30 = [], [], [], []
    # for date in df_final['date'].tolist():
    #     try:
    #         d3.append( df_rsi[df_rsi['DATE'] == date]['RSI_3DAY'].values[0] )
    #         d9.append( df_rsi[df_rsi['DATE'] == date]['RSI_9DAY'].values[0] )
    #         d14.append( df_rsi[df_rsi['DATE'] == date]['RSI_14DAY'].values[0] )
    #         d30.append( df_rsi[df_rsi['DATE'] == date]['RSI_30DAY'].values[0] )
    #     except:
    #         print('invalid rsi date: {}'.format(date))
    # df_final.loc[:,'RSI_3DAY'] = d3
    # df_final.loc[:,'RSI_9DAY'] = d9
    # df_final.loc[:,'RSI_14DAY'] = d14
    # df_final.loc[:,'RSI_30DAY'] = d30


    # ##normalize
    # scaler = MinMaxScaler()
    # for column in df_final.columns:
    #     if column != 'date' and column != 'm_1_close':
    #         df_final[column] = scaler.fit_transform(df_final[[column]])



    # ##ground_truth
    # date_list = df_final['date'].to_list()
    # m1_list = df_final['m_1_close'].to_list()
    # gt_1 = m1_list[1:] + [math.nan]
    # date_1 = date_list[1:] + [math.nan]
    # gt_2 = m1_list[2:] + [math.nan, math.nan]
    # date_2 = date_list[2:] + [math.nan, math.nan]
    # gt_3 = m1_list[3:] + [math.nan, math.nan, math.nan]
    # date_3 = date_list[3:] + [math.nan, math.nan, math.nan]
    # gt_4 = m1_list[4:] + [math.nan, math.nan, math.nan, math.nan]
    # date_4 = date_list[4:] + [math.nan, math.nan, math.nan, math.nan]
    # gt_5 = m1_list[5:] + [math.nan, math.nan, math.nan, math.nan, math.nan]
    # date_5 = date_list[5:] + [math.nan, math.nan, math.nan, math.nan, math.nan]

    # df_final.loc[:,'gt_1'] = gt_1
    # df_final.loc[:,'gt_2'] = gt_2
    # df_final.loc[:,'gt_3'] = gt_3
    # df_final.loc[:,'gt_4'] = gt_4
    # df_final.loc[:,'gt_5'] = gt_5

    # df_final.drop(df_final.tail(5).index, inplace=True)
    # #df_final = df_final.drop(columns=['m_1_close'])
    # df_final = df_final.reset_index(drop=True)
    # print(df_final.tail(10))



    # ##output
    # df_final.to_pickle('./vix_future_preprocessed_aug_1.pkl')
    print('completed')





def main_daily(_date_, vix_table_path='./VIX_2011-2021.xlsx', vvix_table_path='./VVIX_2012-2021.xlsx'):    
    # global variables
    pickle_path = './vix_{}_feature.pkl'.format(_date_)
    df_final = pd.read_pickle(pickle_path)
    year = _date_[:4]
    _month_ = _date_[4:6]
    _day_ = _date_[6:]




    # term structure
    mth_code = {'VXF':1, 'VXG':2, 'VXH':3, 'VXJ':4, 'VXK':5, 'VXM':6, 'VXN':7, 'VXQ':8, 'VXU':9, 'VXV':10, 'VXX':11, 'VXZ':12}
    subdir = year+'/'+_date_ #2011/20110101
    b_dates = pd.bdate_range(year+'/01/01', year+'/12/31')
    date = subdir.split('/')[-1]
    if date in b_dates:
        df_dic = {}
        df_dic['date'] = date
        futures = sorted(os.listdir(subdir)) #VXF1, VXG!, ... 
        curt_mth = int(subdir.split('/')[-1][4:6]) #1
        # print(date, curt_mth, mth_code[futures[0][:3]])
        if curt_mth == 1 and mth_code[futures[0][:3]] == 1:
            for future in futures: #default 8
                    fut_mth = mth_code[future[:3]] 
                    if fut_mth > curt_mth:
                        trm_mth = fut_mth - curt_mth + 1 
                    elif fut_mth == curt_mth:
                        trm_mth = 1
                    if trm_mth >= 9:
                        continue
                    df = pd.read_csv(os.path.join(subdir, future), compression='gzip')
                    intraday_high = df['HighTradePrice'].max()
                    intraday_low = df['LowTradePrice'].min()
                    try:
                        settle_close = [x for x in df['CloseTradePrice'].tolist() if (math.isnan(x) == False)][-1]
                    except:
                        settle_close = math.nan   
                    df_dic['m_{}_close'.format(trm_mth)] = settle_close
                    df_dic['m_{}_hl'.format(trm_mth)] = intraday_high - intraday_low  
                    df_dic['m_{}_high'.format(trm_mth)] = intraday_high
                    df_dic['m_{}_low'.format(trm_mth)] = intraday_low
        elif curt_mth < 5 and mth_code[futures[0][:3]] != 1:
            if mth_code[futures[0][:3]] == curt_mth:
                for future in futures: #default 8
                    fut_mth = mth_code[future[:3]] 
                    if fut_mth > curt_mth:
                        trm_mth = fut_mth - curt_mth + 1 
                    elif fut_mth == curt_mth:
                        trm_mth = 1
                    else:
                        trm_mth = fut_mth + 13 - curt_mth
                    if trm_mth >= 9:
                        continue
                    df = pd.read_csv(os.path.join(subdir, future), compression='gzip')
                    intraday_high = df['HighTradePrice'].max()
                    intraday_low = df['LowTradePrice'].min()
                    try:
                        settle_close = [x for x in df['CloseTradePrice'].tolist() if (math.isnan(x) == False)][-1]
                    except:
                        settle_close = math.nan   
                    df_dic['m_{}_close'.format(trm_mth)] = settle_close
                    df_dic['m_{}_hl'.format(trm_mth)] = intraday_high - intraday_low  
                    df_dic['m_{}_high'.format(trm_mth)] = intraday_high
                    df_dic['m_{}_low'.format(trm_mth)] = intraday_low
            else:
                for future in futures:
                    fut_mth = mth_code[future[:3]] 
                    if fut_mth > curt_mth:
                        trm_mth = fut_mth - curt_mth
                    else:
                        trm_mth = fut_mth + 13 - curt_mth
                    if trm_mth >= 9:
                        continue
                    df = pd.read_csv(os.path.join(subdir, future), compression='gzip')
                    intraday_high = df['HighTradePrice'].max()
                    intraday_low = df['LowTradePrice'].min()
                    try:
                        settle_close = [x for x in df['CloseTradePrice'].tolist() if (math.isnan(x) == False)][-1]
                    except:
                        settle_close = math.nan     
                    df_dic['m_{}_close'.format(trm_mth)] = settle_close
                    df_dic['m_{}_hl'.format(trm_mth)] = intraday_high - intraday_low  
                    df_dic['m_{}_high'.format(trm_mth)] = intraday_high
                    df_dic['m_{}_low'.format(trm_mth)] = intraday_low
        else:
            if curt_mth in [mth_code[x[:3]] for x in futures]:
                for future in futures:
                    fut_mth = mth_code[future[:3]] 
                    if fut_mth < curt_mth:
                        trm_mth = fut_mth + 13 - curt_mth
                    else:
                        trm_mth = fut_mth + 1 - curt_mth
                    if trm_mth >= 9:
                        continue
                    df = pd.read_csv(os.path.join(subdir, future), compression='gzip')
                    intraday_high = df['HighTradePrice'].max()
                    intraday_low = df['LowTradePrice'].min()
                    try:
                        settle_close = [x for x in df['CloseTradePrice'].tolist() if (math.isnan(x) == False)][-1]
                    except:
                        settle_close = math.nan   
                    df_dic['m_{}_close'.format(trm_mth)] = settle_close
                    df_dic['m_{}_hl'.format(trm_mth)] = intraday_high - intraday_low  
                    df_dic['m_{}_high'.format(trm_mth)] = intraday_high
                    df_dic['m_{}_low'.format(trm_mth)] = intraday_low
            else:
                for future in futures:
                    fut_mth = mth_code[future[:3]] 
                    if fut_mth < curt_mth:
                        trm_mth = fut_mth + 12 - curt_mth
                    else:
                        trm_mth = fut_mth - curt_mth
                    if trm_mth >= 9:
                        continue
                    df = pd.read_csv(os.path.join(subdir, future), compression='gzip')
                    intraday_high = df['HighTradePrice'].max()
                    intraday_low = df['LowTradePrice'].min()
                    try:
                        settle_close = [x for x in df['CloseTradePrice'].tolist() if (math.isnan(x) == False)][-1]
                    except:
                        settle_close = math.nan   
                    df_dic['m_{}_close'.format(trm_mth)] = settle_close
                    df_dic['m_{}_hl'.format(trm_mth)] = intraday_high - intraday_low  
                    df_dic['m_{}_high'.format(trm_mth)] = intraday_high
                    df_dic['m_{}_low'.format(trm_mth)] = intraday_low               
    else:
        print('the input date is invalid') ##################################terminate!!!!!!!!!!!!!!!!!!!!!!
    
    pairs = list(itertools.combinations([8,7,6,5,4,3,2,1], 2))
    for pair in pairs:
        df_dic['m_{}-{}'.format(pair[0],pair[1])] = df_dic['m_{}_close'.format(pair[0])] - df_dic['m_{}_close'.format(pair[1])] 
    
    for i in range(2,9):
        del df_dic['m_{}_close'.format(i)]
        del df_dic['m_{}_high'.format(i)]
        del df_dic['m_{}_low'.format(i)]

    df_final = df_final.append(df_dic, ignore_index=True) ###############################check!!!!!!!!!!!!!!!!!!!!!!





    # technical variables for 1mth vix future AND vix OR/AND vvix
    xls = pd.ExcelFile(vix_table_path)
    df_vix = pd.read_excel(xls, header=0)
    
    def prior_avg(datafarme, column, date, day):
        idx = datafarme[datafarme['date'] == date].index[0]
        sum_ = 0
        for i in range(idx-day+1,idx+1):
            sum_ += datafarme.loc[i][column]
        avg = sum_/day
        return avg
    
    def avg_move(datafarme, column, date, day, type_):
        avg = prior_avg(datafarme, column, date, day)
        spot = datafarme[datafarme['date'] == date][column].values[0]
        if type_ == 'above':
            result = spot - avg > 0
            return 1 if result else 0
        elif type_ == 'below':
            result = spot - avg < 0
            return 1 if result else 0
            
    def avg_three_day_move(datafarme, column, date, day, type_):
        idx = datafarme.index[datafarme['date']==date].tolist()[0]
        if type_ == 'above':
            d1 = avg_move(datafarme, column, datafarme.iloc[idx-1]['date'], day, type_)
            d2 = avg_move(datafarme, column, datafarme.iloc[idx-2]['date'], day, type_)
            d3 = avg_move(datafarme, column, datafarme.iloc[idx-3]['date'], day, type_)
            if d1 == d2 == d3 == 1:
                return 1
            else:
                return 0
        elif type_ == 'below':
            d1 = avg_move(datafarme, column, datafarme.iloc[idx-1]['date'], day, type_)
            d2 = avg_move(datafarme, column, datafarme.iloc[idx-2]['date'], day, type_)
            d3 = avg_move(datafarme, column, datafarme.iloc[idx-3]['date'], day, type_)
            if d1 == d2 == d3 == 1:
                return 1
            else:
                return 0

    def bollinger_bands_move(datafarme, column, date, type_):
        spot = datafarme[datafarme['date'] == date][column].values[0]
        avg = prior_avg(datafarme, column, date, 21)
        idx = datafarme.index[datafarme['date']==date].tolist()[0]
        std = datafarme.iloc[idx-21:idx][column].std()
        if type_ == 'above':
            return 1 if spot > avg+2*std else 0
        elif type_ == 'below':
            return 1 if spot < avg-2*std else 0

    def exponential_avg(datafarme, column, date, day):
        idx = datafarme.index[datafarme['date']==date].tolist()[0]
        avg_t_1 = prior_avg(datafarme, column, datafarme.iloc[idx-1]['date'], day)
        spot = datafarme[datafarme['date'] == date][column].values[0]
        exp = (spot - avg_t_1) * 2 / (day+1) + avg_t_1
        return exp

    def exponential_avg_move(datafarme, column, date, day, type_):
        exp_avg = exponential_avg(datafarme, column, date, day)
        spot = datafarme[datafarme['date'] == date][column].values[0]
        if type_ == 'above':
            result = spot - exp_avg > 0
            return 1 if result else 0
        elif type_ == 'below':
            result = spot - exp_avg < 0
            return 1 if result else 0

    def disparity(datafarme, column, date, day):
        avg = prior_avg(datafarme, column, date, day)
        spot = datafarme[datafarme['date'] == date][column].values[0]
        return spot / avg * 100
    
    def disparity_move(datafarme, column, date, day, type_):
        disp_today = disparity(datafarme, column, date, day)
        idx = datafarme.index[datafarme['date']==date].tolist()[0]
        disp_yesterday = disparity(datafarme, column, datafarme.iloc[idx-1]['date'], day)
        if type_ == 'above':
            result = disp_today - disp_yesterday > 0
            return 1 if result else 0
        elif type_ == 'below':
            result = disp_today - disp_yesterday < 0
            return 1 if result else 0

    def momentum1(datafarme, column, date):
        spot = datafarme[datafarme['date'] == date][column].values[0]
        idx = datafarme.index[datafarme['date']==date].tolist()[0]
        prior = datafarme.iloc[idx-5][column]#.values[0]
        return spot / prior * 100

    def momentum2(datafarme, column, date):
        spot = datafarme[datafarme['date'] == date][column].values[0]
        idx = datafarme.index[datafarme['date']==date].tolist()[0]
        prior = datafarme.iloc[idx-5][column]#.values[0]
        return (spot - prior + 0.001) * 100

    def ROC(datafarme, column, date):
        spot = datafarme[datafarme['date'] == date][column].values[0]
        mtm = momentum2(datafarme, column, date)
        return spot / mtm * 100

    def ROC_move(datafarme, column, date, type_):
        roc_today = ROC(datafarme, column, date)
        idx = datafarme.index[datafarme['date']==date].tolist()[0]
        roc_yesterday = ROC(datafarme, column, datafarme.iloc[idx-1]['date'])
        if type_ == 'above':
            result = roc_today - roc_yesterday > 0
            return 1 if result else 0
        elif type_ == 'below':
            result = roc_today - roc_yesterday < 0
            return 1 if result else 0

    def stochastic_william(datafarme, column, date, day=14, type_='william'):
        idx = datafarme[datafarme['date'] == date].index[0]
        spot = datafarme[datafarme['date'] == date][column+'close'].values[0]
        highest_high = -99
        lowest_low = 99
        for i in range(idx-day+1,idx+1):
            if datafarme.loc[i][column+'high'] > highest_high:
                highest_high = datafarme.loc[i][column+'high']
            if datafarme.loc[i][column+'low'] < lowest_low:
                lowest_low = datafarme.loc[i][column+'low']
        if type_ == 'william':
            return (highest_high-spot) / (highest_high-lowest_low)
        elif type_ == 'stochastic':
            return (spot-lowest_low) / (highest_high-lowest_low) * 100
            

    date = _date_
    #moving average
    vix_p_3_avg = prior_avg(df_vix, 'close', date, 3)
    vix_p_5_avg = prior_avg(df_vix, 'close', date, 5)
    vix_p_10_avg = prior_avg(df_vix, 'close', date, 10)

    fut_p_3_avg = prior_avg(df_final, 'm_1_close', date, 3)
    fut_p_5_avg = prior_avg(df_final, 'm_1_close', date, 5)
    fut_p_10_avg = prior_avg(df_final, 'm_1_close', date, 10)
    
    #moving average move
    vix_p_3_above = avg_move(df_vix, 'close', date, 3, 'above')
    vix_p_5_above = avg_move(df_vix, 'close', date, 5, 'above')
    vix_p_10_above = avg_move(df_vix, 'close', date, 10, 'above')
    vix_p_3_below = avg_move(df_vix, 'close', date, 3, 'below')
    vix_p_5_below = avg_move(df_vix, 'close', date, 5, 'below')
    vix_p_10_below = avg_move(df_vix, 'close', date, 10, 'below')
    vix_p_3_above_3d = avg_three_day_move(df_vix, 'close', date, 3, 'above')
    vix_p_5_above_3d = avg_three_day_move(df_vix, 'close', date, 5, 'above')
    vix_p_10_above_3d = avg_three_day_move(df_vix, 'close', date, 10, 'above')
    vix_p_3_below_3d = avg_three_day_move(df_vix, 'close', date, 3, 'below')
    vix_p_5_below_3d = avg_three_day_move(df_vix, 'close', date, 5, 'below')
    vix_p_10_below_3d = avg_three_day_move(df_vix, 'close', date, 10, 'below')

    fut_p_3_above = avg_move(df_final, 'm_1_close', date, 3, 'above')
    fut_p_5_above = avg_move(df_final, 'm_1_close', date, 5, 'above')
    fut_p_10_above = avg_move(df_final, 'm_1_close', date, 10, 'above')
    fut_p_3_below = avg_move(df_final, 'm_1_close', date, 3, 'below')
    fut_p_5_below = avg_move(df_final, 'm_1_close', date, 5, 'below')
    fut_p_10_below = avg_move(df_final, 'm_1_close', date, 10, 'below')
    fut_p_3_above_3d = avg_three_day_move(df_final, 'm_1_close', date, 3, 'above')
    fut_p_5_above_3d = avg_three_day_move(df_final, 'm_1_close', date, 5, 'above')
    fut_p_10_above_3d = avg_three_day_move(df_final, 'm_1_close', date, 10, 'above')
    fut_p_3_below_3d = avg_three_day_move(df_final, 'm_1_close', date, 3, 'below')
    fut_p_5_below_3d = avg_three_day_move(df_final, 'm_1_close', date, 5, 'below')
    fut_p_10_below_3d = avg_three_day_move(df_final, 'm_1_close', date, 10, 'below')


    #bollinger band move
    vix_bollinger_bands_above = bollinger_bands_move(df_vix, 'close', date, 'above')
    vix_bollinger_bands_below = bollinger_bands_move(df_vix, 'close', date, 'below')

    fut_bollinger_bands_above = bollinger_bands_move(df_final, 'm_1_close', date, 'above')
    fut_bollinger_bands_below = bollinger_bands_move(df_final, 'm_1_close', date, 'below')


    #exponential moving average
    vix_p_3_exp_avg = exponential_avg(df_vix, 'close', date, 3)
    vix_p_5_exp_avg = exponential_avg(df_vix, 'close', date, 5)
    vix_p_10_exp_avg = exponential_avg(df_vix, 'close', date, 10)
    
    fut_p_3_exp_avg = exponential_avg(df_final, 'm_1_close', date, 3)
    fut_p_5_exp_avg = exponential_avg(df_final, 'm_1_close', date, 5)
    fut_p_10_exp_avg = exponential_avg(df_final, 'm_1_close', date, 10)
    

    #exponential moving average move
    vix_p_3_exp_above = exponential_avg_move(df_vix, 'close', date, 3, 'above')
    vix_p_5_exp_above = exponential_avg_move(df_vix, 'close', date, 5, 'above')
    vix_p_10_exp_above = exponential_avg_move(df_vix, 'close', date, 10, 'above')
    vix_p_3_exp_below = exponential_avg_move(df_vix, 'close', date, 3, 'below')
    vix_p_5_exp_below = exponential_avg_move(df_vix, 'close', date, 5, 'below')
    vix_p_10_exp_below = exponential_avg_move(df_vix, 'close', date, 10, 'below')
    
    fut_p_3_exp_above = exponential_avg_move(df_final, 'm_1_close', date, 3, 'above')
    fut_p_5_exp_above = exponential_avg_move(df_final, 'm_1_close', date, 5, 'above')
    fut_p_10_exp_above = exponential_avg_move(df_final, 'm_1_close', date, 10, 'above')
    fut_p_3_exp_below = exponential_avg_move(df_final, 'm_1_close', date, 3, 'below')
    fut_p_5_exp_below = exponential_avg_move(df_final, 'm_1_close', date, 5, 'below')
    fut_p_10_exp_below = exponential_avg_move(df_final, 'm_1_close', date, 10, 'below')
    
    
    #disparity
    vix_disp_3 = disparity(df_vix, 'close', date, 3)
    vix_disp_5 = disparity(df_vix, 'close', date, 5)
    
    fut_disp_3 = disparity(df_final, 'm_1_close', date, 3)
    fut_disp_5 = disparity(df_final, 'm_1_close', date, 5)


    #disparity move
    vix_disp_3_above = disparity_move(df_vix, 'close', date, 3, 'above')
    vix_disp_5_above = disparity_move(df_vix, 'close', date, 5, 'above')
    vix_disp_3_below = disparity_move(df_vix, 'close', date, 3, 'below')
    vix_disp_5_below = disparity_move(df_vix, 'close', date, 5, 'below')
    
    fut_disp_3_above = disparity_move(df_final, 'm_1_close', date, 3, 'above')
    fut_disp_5_above = disparity_move(df_final, 'm_1_close', date, 5, 'above')
    fut_disp_3_below = disparity_move(df_final, 'm_1_close', date, 3, 'below')
    fut_disp_5_below = disparity_move(df_final, 'm_1_close', date, 5, 'below')


    #momentum1
    vix_mom1_5 = momentum1(df_vix, 'close', date)
    fut_mom1_5 = momentum1(df_final, 'm_1_close', date)


    #momentum2
    vix_mom2_5 = momentum2(df_vix, 'close', date)
    fut_mom2_5 = momentum2(df_final, 'm_1_close', date)


    #ROC
    vix_roc = ROC(df_vix, 'close', date)
    fut_roc = ROC(df_final, 'm_1_close', date)
    

    #ROC move
    vix_roc_above = ROC_move(df_vix, 'close', date, 'above')
    vix_roc_below = ROC_move(df_vix, 'close', date, 'below')
    
    fut_roc_above = ROC_move(df_final, 'm_1_close', date, 'above')
    fut_roc_below = ROC_move(df_final, 'm_1_close', date, 'below')


    #william %R
    vix_william = stochastic_william(df_vix, '', date, day=14, type_='william')
    fut_william = stochastic_william(df_final, 'm_1_', date, day=14, type_='william')
    
    #stochastic oscillator
    vix_stochastic = stochastic_william(df_vix, '', date, day=14, type_='stochastic')
    fut_stochastic = stochastic_william(df_final, 'm_1_', date, day=14, type_='stochastic')
        
    
    new_data_idx = df_final[df_final['date'] == _date_].index[0]

    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_3_avg')] = vix_p_3_avg 
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_5_avg')] = vix_p_5_avg
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_10_avg')] = vix_p_10_avg
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_3_above')] = vix_p_3_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_5_above')] = vix_p_5_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_10_above')] = vix_p_10_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_3_below')] = vix_p_3_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_5_below')] = vix_p_5_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_10_below')] = vix_p_10_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_3_avg')] = fut_p_3_avg
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_5_avg')] = fut_p_5_avg
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_10_avg')] = fut_p_10_avg
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_3_above')] = fut_p_3_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_5_above')] = fut_p_5_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_10_above')] = fut_p_10_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_3_below')] = fut_p_3_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_5_below')] = fut_p_5_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_10_below')] = fut_p_10_below
    
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_3_3day_above')] = vix_p_3_above_3d
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_5_3day_above')] = vix_p_5_above_3d
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_10_3day_above')] = vix_p_10_above_3d
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_3_3day_below')] = vix_p_3_below_3d
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_5_3day_below')] = vix_p_5_below_3d
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_10_3day_below')] = vix_p_10_below_3d
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_3_3day_above')] = fut_p_3_above_3d
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_5_3day_above')] = fut_p_5_above_3d
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_10_3day_above')] = fut_p_10_above_3d
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_3_3day_below')] = fut_p_3_below_3d
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_5_3day_below')] = fut_p_5_below_3d
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_10_3day_below')] = fut_p_10_below_3d

    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_3_exp_avg')] = vix_p_3_exp_avg
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_5_exp_avg')] = vix_p_5_exp_avg
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_10_exp_avg')] = vix_p_10_exp_avg
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_3_exp_above')] = vix_p_3_exp_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_5_exp_above')] = vix_p_5_exp_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_10_exp_above')] = vix_p_10_exp_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_3_exp_below')] = vix_p_3_exp_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_5_exp_below')] = vix_p_5_exp_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_p_10_exp_below')] = vix_p_10_exp_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_3_exp_avg')] = fut_p_3_exp_avg
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_5_exp_avg')] = fut_p_5_exp_avg
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_10_exp_avg')] = fut_p_10_exp_avg
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_3_exp_above')] = fut_p_3_exp_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_5_exp_above')] = fut_p_5_exp_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_10_exp_above')] = fut_p_10_exp_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_3_exp_below')] = fut_p_3_exp_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_5_exp_below')] = fut_p_5_exp_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_p_10_exp_below')] = fut_p_10_exp_below

    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_BB_above')] = vix_bollinger_bands_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_BB_below')] = vix_bollinger_bands_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_BB_above')] = fut_bollinger_bands_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_BB_below')] = fut_bollinger_bands_below

    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_disp3_avg')] = vix_disp_3
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_disp5_avg')] = vix_disp_5
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_disp3_above')] = vix_disp_3_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_disp5_above')] = vix_disp_5_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_disp3_below')] = vix_disp_3_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_disp5_below')] = vix_disp_5_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_disp3_avg')] = fut_disp_3
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_disp5_avg')] = fut_disp_5
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_disp3_above')] = fut_disp_3_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_disp5_above')] = fut_disp_5_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_disp3_below')] = fut_disp_3_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_disp5_below')] = fut_disp_5_below

    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_momentum1')] = vix_mom1_5
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_momentum2')] = vix_mom2_5
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_roc')] = vix_roc
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_roc_above')] = vix_roc_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_roc_below')] = vix_roc_below
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_momentum1')] = fut_mom1_5
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_momentum2')] = fut_mom2_5
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_roc')] = fut_roc
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_roc_above')] = fut_roc_above
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_roc_below')] = fut_roc_below

    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_william')] = vix_william
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vix_stochastic')] = vix_stochastic
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_william')] = fut_william
    df_final.iloc[new_data_idx, df_final.columns.get_loc('fut_stochastic')] = fut_stochastic
    



    ##vvix
    df_vvix = pd.read_excel(xls, header=0)
    df_hl = df_vvix[['high date', 'VVIX high', 'VVIX low']]
    df_close = df_vvix[['close date', 'VVIX close']]
    
    vvix_hl = df_hl[df_hl['high date'] == _date_]['VVIX high'].values[0] - df_hl[df_hl['high date'] == _date_]['VVIX low'].values[0]
    vvix = df_close[df_close['close date'] == _date_]['VVIX close'].values[0]
    
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vvix')] = vvix
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vvix_hl')] = vvix_hl




    #rsi
    xls = pd.ExcelFile('./RSI_data.xlsx')
    df_rsi = pd.read_excel(xls)

    d3 = df_rsi[df_rsi['DATE'] == _date_]['RSI_3DAY'].values[0] 
    d9 = df_rsi[df_rsi['DATE'] == _date_]['RSI_9DAY'].values[0] 
    d14 = df_rsi[df_rsi['DATE'] == _date_]['RSI_14DAY'].values[0] 
    d30 = df_rsi[df_rsi['DATE'] == _date_]['RSI_30DAY'].values[0] 

    df_final.iloc[new_data_idx, df_final.columns.get_loc('RSI_3DAY')] = d3
    df_final.iloc[new_data_idx, df_final.columns.get_loc('RSI_9DAY')] = d9
    df_final.iloc[new_data_idx, df_final.columns.get_loc('RSI_14DAY')] = d14
    df_final.iloc[new_data_idx, df_final.columns.get_loc('RSI_30DAY')] = d30




    ##normalize
    scaler = MinMaxScaler()
    for column in df_final.columns:
        if column != 'date' and column != 'm_1_close':
            df_final[column] = scaler.fit_transform(df_final[[column]])
    



    ##feature output
    df_final.to_pickle('./vix_{}_feature.pkl'.format(_date_))
    print('features saving completed')



    ##ground_truth
    date_list = df_final['date'].to_list()
    m1_list = df_final['m_1_close'].to_list()
    gt_1 = m1_list[1:] + [math.nan]
    date_1 = date_list[1:] + [math.nan]
    gt_2 = m1_list[2:] + [math.nan, math.nan]
    date_2 = date_list[2:] + [math.nan, math.nan]
    gt_3 = m1_list[3:] + [math.nan, math.nan, math.nan]
    date_3 = date_list[3:] + [math.nan, math.nan, math.nan]
    gt_4 = m1_list[4:] + [math.nan, math.nan, math.nan, math.nan]
    date_4 = date_list[4:] + [math.nan, math.nan, math.nan, math.nan]
    gt_5 = m1_list[5:] + [math.nan, math.nan, math.nan, math.nan, math.nan]
    date_5 = date_list[5:] + [math.nan, math.nan, math.nan, math.nan, math.nan]

    df_final.loc[:,'gt_1'] = gt_1
    df_final.loc[:,'gt_2'] = gt_2
    df_final.loc[:,'gt_3'] = gt_3
    df_final.loc[:,'gt_4'] = gt_4
    df_final.loc[:,'gt_5'] = gt_5

    df_final.drop(df_final.tail(5).index, inplace=True)
    #df_final = df_final.drop(columns=['m_1_close'])
    df_final = df_final.reset_index(drop=True)
    print(df_final.tail(10))



    ##output
    df_final.to_pickle('./vix_{}_feature_gt.pkl'.format(_date_))
    print('features & ground_truths completed')



if __name__ == "__main__":
    # main()
    main_new()
    # all_data = pd.read_pickle('./vix_future_preprocessed_aug_1.pkl')
    # print(all_data)
    # print('\t')
    # _date_ = '20210526'
    # idx = all_data[all_data['date'] == _date_].index[0]
    # all_data.iloc[idx, all_data.columns.get_loc('gt_5')] = 9999
    # print(all_data)
    # print(all_data.index[all_data['date'] == '20170103'])
    # print(all_data.index[all_data['date'] == '20171229'])
    # print(all_data.iloc[1194:1445]['date'].tolist())
    # print(all_data.iloc[1194:1445]['m_1_close'].tolist())


