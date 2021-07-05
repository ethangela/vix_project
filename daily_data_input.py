# -*- coding: utf-8 -*-
import pandas as pd
from collections import defaultdict
import itertools
from sklearn.preprocessing import MinMaxScaler
import os
import math
import numpy as np
from daily_configs import get_config
import pickle

def main_backup(config): 
    if os.path.isdir(config.pickle_dir) is False:
        os.mkdir(config.pickle_dir)

    # term structure
    print('Phase 1 started ...')
    mth_code = {'VXF':1, 'VXG':2, 'VXH':3, 'VXJ':4, 'VXK':5, 'VXM':6, 'VXN':7, 'VXQ':8, 'VXU':9, 'VXV':10, 'VXX':11, 'VXZ':12}
    pairs = list(itertools.combinations([8,7,6,5,4,3,2,1], 2))
    c = 0
    start_year = int(config.start_year)
    current_year = int(config.date[:4])
    years_list = [str(year) for year in range(start_year, current_year+1)]

    for year in years_list:
        subdirs = [ f.path for f in os.scandir(year+'/') if f.is_dir() ] #2011/20110101, 2011/20110102, ...
        subdirs = sorted(subdirs)
        b_dates = pd.bdate_range(year+'/01/01', year+'/12/31')
        for i,subdir in enumerate(subdirs): #0, 2011/20110101 ...
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
                
                if c == 0:
                    df_final = pd.DataFrame([df_dic])
                else:
                    df_final = df_final.append(df_dic, ignore_index=True)               
                c += 1
            else:
                continue
    for pair in pairs:
        df_final.loc[:,'m_{}-{}'.format(pair[0],pair[1])] = df_final['m_{}_close'.format(pair[0])] - df_final['m_{}_close'.format(pair[1])] 
    for i in range(2,9):
        df_final = df_final.drop(columns=['m_{}_close'.format(i)])
        df_final = df_final.drop(columns=['m_{}_high'.format(i)])
        df_final = df_final.drop(columns=['m_{}_low'.format(i)])
    df_final = df_final.fillna(method='ffill')
    print('Phase 1 completed. Samples shown below:')
    print(df_final.tail(5))
    df_final.to_pickle(os.path.join(config.pickle_dir, 'vix_future_backup_phase_1.pkl'))
    




    # #technical variables for 1mth vix future AND vix OR/AND vvix
    print('Phase 2 started ...')
    df_final = pd.read_pickle(os.path.join(config.pickle_dir, 'vix_future_backup_phase_1.pkl'))
    xls = pd.ExcelFile(config.vix_table_path)
    df_vix = pd.read_excel(xls, header=0)
    invalid_date_due_to_vix = []
    for date in df_final['date'].tolist():
        if len(df_vix[df_vix['date'] == date]['close'].values) != 0:
            continue
        else:
            invalid_date_due_to_vix.append(date)
            df_final = df_final.drop(df_final[df_final['date'] == date].index)
    df_final = df_final.reset_index(drop=True)
    print('invalid_date_due_to_vix: {}'.format(invalid_date_due_to_vix))

    
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
            

    vix_p_3_avg, vix_p_5_avg, vix_p_10_avg = [], [], []
    vix_p_3_above, vix_p_5_above, vix_p_10_above = [], [], []
    vix_p_3_below, vix_p_5_below, vix_p_10_below = [], [], []
    vix_p_3_above_3d, vix_p_5_above_3d, vix_p_10_above_3d = [], [], []
    vix_p_3_below_3d, vix_p_5_below_3d, vix_p_10_below_3d = [], [], []
    vix_bollinger_bands_above, vix_bollinger_bands_below = [], []
    vix_p_3_exp_avg, vix_p_5_exp_avg, vix_p_10_exp_avg = [], [], []
    vix_p_3_exp_above, vix_p_5_exp_above, vix_p_10_exp_above = [], [], []
    vix_p_3_exp_below, vix_p_5_exp_below, vix_p_10_exp_below = [], [], []
    vix_disp_3, vix_disp_5 = [], []
    vix_disp_3_above, vix_disp_5_above = [], []
    vix_disp_3_below, vix_disp_5_below = [], []
    vix_mom1_5, vix_mom2_5 = [], []
    vix_roc, vix_roc_above, vix_roc_below = [], [], []
    vix_william = []
    vix_stochastic = []


    fut_p_3_avg, fut_p_5_avg, fut_p_10_avg = [], [], []
    fut_p_3_above, fut_p_5_above, fut_p_10_above = [], [], []
    fut_p_3_below, fut_p_5_below, fut_p_10_below = [], [], []
    fut_p_3_above_3d, fut_p_5_above_3d, fut_p_10_above_3d = [], [], []
    fut_p_3_below_3d, fut_p_5_below_3d, fut_p_10_below_3d = [], [], []
    fut_bollinger_bands_above, fut_bollinger_bands_below = [], []
    fut_p_3_exp_avg, fut_p_5_exp_avg, fut_p_10_exp_avg = [], [], []
    fut_p_3_exp_above, fut_p_5_exp_above, fut_p_10_exp_above = [], [], []
    fut_p_3_exp_below, fut_p_5_exp_below, fut_p_10_exp_below = [], [], []
    fut_disp_3, fut_disp_5 = [], []
    fut_disp_3_above, fut_disp_5_above = [], []
    fut_disp_3_below, fut_disp_5_below = [], []
    fut_mom1_5, fut_mom2_5 = [], []
    fut_roc, fut_roc_above, fut_roc_below = [], [], []
    fut_william = []
    fut_stochastic = []


    for date in df_final['date'].tolist()[20:]: 
        #moving average
        vix_p_3_avg.append(prior_avg(df_vix, 'close', date, 3))
        vix_p_5_avg.append(prior_avg(df_vix, 'close', date, 5))
        vix_p_10_avg.append(prior_avg(df_vix, 'close', date, 10))

        fut_p_3_avg.append(prior_avg(df_final, 'm_1_close', date, 3))
        fut_p_5_avg.append(prior_avg(df_final, 'm_1_close', date, 5))
        fut_p_10_avg.append(prior_avg(df_final, 'm_1_close', date, 10))
        
        #moving average move
        vix_p_3_above.append(avg_move(df_vix, 'close', date, 3, 'above'))
        vix_p_5_above.append(avg_move(df_vix, 'close', date, 5, 'above'))
        vix_p_10_above.append(avg_move(df_vix, 'close', date, 10, 'above'))
        vix_p_3_below.append(avg_move(df_vix, 'close', date, 3, 'below'))
        vix_p_5_below.append(avg_move(df_vix, 'close', date, 5, 'below'))
        vix_p_10_below.append(avg_move(df_vix, 'close', date, 10, 'below'))
        vix_p_3_above_3d.append(avg_three_day_move(df_vix, 'close', date, 3, 'above'))
        vix_p_5_above_3d.append(avg_three_day_move(df_vix, 'close', date, 5, 'above'))
        vix_p_10_above_3d.append(avg_three_day_move(df_vix, 'close', date, 10, 'above'))
        vix_p_3_below_3d.append(avg_three_day_move(df_vix, 'close', date, 3, 'below'))
        vix_p_5_below_3d.append(avg_three_day_move(df_vix, 'close', date, 5, 'below'))
        vix_p_10_below_3d.append(avg_three_day_move(df_vix, 'close', date, 10, 'below'))

        fut_p_3_above.append(avg_move(df_final, 'm_1_close', date, 3, 'above'))
        fut_p_5_above.append(avg_move(df_final, 'm_1_close', date, 5, 'above'))
        fut_p_10_above.append(avg_move(df_final, 'm_1_close', date, 10, 'above'))
        fut_p_3_below.append(avg_move(df_final, 'm_1_close', date, 3, 'below'))
        fut_p_5_below.append(avg_move(df_final, 'm_1_close', date, 5, 'below'))
        fut_p_10_below.append(avg_move(df_final, 'm_1_close', date, 10, 'below'))
        fut_p_3_above_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 3, 'above'))
        fut_p_5_above_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 5, 'above'))
        fut_p_10_above_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 10, 'above'))
        fut_p_3_below_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 3, 'below'))
        fut_p_5_below_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 5, 'below'))
        fut_p_10_below_3d.append(avg_three_day_move(df_final, 'm_1_close', date, 10, 'below'))


        #bollinger band move
        vix_bollinger_bands_above.append(bollinger_bands_move(df_vix, 'close', date, 'above'))
        vix_bollinger_bands_below.append(bollinger_bands_move(df_vix, 'close', date, 'below'))

        fut_bollinger_bands_above.append(bollinger_bands_move(df_final, 'm_1_close', date, 'above'))
        fut_bollinger_bands_below.append(bollinger_bands_move(df_final, 'm_1_close', date, 'below'))


        #exponential moving average
        vix_p_3_exp_avg.append(exponential_avg(df_vix, 'close', date, 3))
        vix_p_5_exp_avg.append(exponential_avg(df_vix, 'close', date, 5))
        vix_p_10_exp_avg.append(exponential_avg(df_vix, 'close', date, 10))
        
        fut_p_3_exp_avg.append(exponential_avg(df_final, 'm_1_close', date, 3))
        fut_p_5_exp_avg.append(exponential_avg(df_final, 'm_1_close', date, 5))
        fut_p_10_exp_avg.append(exponential_avg(df_final, 'm_1_close', date, 10))
        

        #exponential moving average move
        vix_p_3_exp_above.append(exponential_avg_move(df_vix, 'close', date, 3, 'above'))
        vix_p_5_exp_above.append(exponential_avg_move(df_vix, 'close', date, 5, 'above'))
        vix_p_10_exp_above.append(exponential_avg_move(df_vix, 'close', date, 10, 'above'))
        vix_p_3_exp_below.append(exponential_avg_move(df_vix, 'close', date, 3, 'below'))
        vix_p_5_exp_below.append(exponential_avg_move(df_vix, 'close', date, 5, 'below'))
        vix_p_10_exp_below.append(exponential_avg_move(df_vix, 'close', date, 10, 'below'))
        
        fut_p_3_exp_above.append(exponential_avg_move(df_final, 'm_1_close', date, 3, 'above'))
        fut_p_5_exp_above.append(exponential_avg_move(df_final, 'm_1_close', date, 5, 'above'))
        fut_p_10_exp_above.append(exponential_avg_move(df_final, 'm_1_close', date, 10, 'above'))
        fut_p_3_exp_below.append(exponential_avg_move(df_final, 'm_1_close', date, 3, 'below'))
        fut_p_5_exp_below.append(exponential_avg_move(df_final, 'm_1_close', date, 5, 'below'))
        fut_p_10_exp_below.append(exponential_avg_move(df_final, 'm_1_close', date, 10, 'below'))
        
        
        #disparity
        vix_disp_3.append(disparity(df_vix, 'close', date, 3))
        vix_disp_5.append(disparity(df_vix, 'close', date, 5))
        
        fut_disp_3.append(disparity(df_final, 'm_1_close', date, 3))
        fut_disp_5.append(disparity(df_final, 'm_1_close', date, 5))


        #disparity move
        vix_disp_3_above.append(disparity_move(df_vix, 'close', date, 3, 'above'))
        vix_disp_5_above.append(disparity_move(df_vix, 'close', date, 5, 'above'))
        vix_disp_3_below.append(disparity_move(df_vix, 'close', date, 3, 'below'))
        vix_disp_5_below.append(disparity_move(df_vix, 'close', date, 5, 'below'))
        
        fut_disp_3_above.append(disparity_move(df_final, 'm_1_close', date, 3, 'above'))
        fut_disp_5_above.append(disparity_move(df_final, 'm_1_close', date, 5, 'above'))
        fut_disp_3_below.append(disparity_move(df_final, 'm_1_close', date, 3, 'below'))
        fut_disp_5_below.append(disparity_move(df_final, 'm_1_close', date, 5, 'below'))


        #momentum1
        vix_mom1_5.append(momentum1(df_vix, 'close', date))
        fut_mom1_5.append(momentum1(df_final, 'm_1_close', date))


        #momentum2
        vix_mom2_5.append(momentum2(df_vix, 'close', date))
        fut_mom2_5.append(momentum2(df_final, 'm_1_close', date))


        #ROC
        vix_roc.append(ROC(df_vix, 'close', date))
        fut_roc.append(ROC(df_final, 'm_1_close', date))
        

        #ROC move
        vix_roc_above.append(ROC_move(df_vix, 'close', date, 'above'))
        vix_roc_below.append(ROC_move(df_vix, 'close', date, 'below'))
        
        fut_roc_above.append(ROC_move(df_final, 'm_1_close', date, 'above'))
        fut_roc_below.append(ROC_move(df_final, 'm_1_close', date, 'below'))


        #william %R
        vix_william.append(stochastic_william(df_vix, '', date, day=14, type_='william'))
        fut_william.append(stochastic_william(df_final, 'm_1_', date, day=14, type_='william'))
        
        #stochastic oscillator
        vix_stochastic.append(stochastic_william(df_vix, '', date, day=14, type_='stochastic'))
        fut_stochastic.append(stochastic_william(df_final, 'm_1_', date, day=14, type_='stochastic'))
        

    df_final = df_final.iloc[20:]
    df_final = df_final.reset_index(drop=True)

    df_final.loc[:,'vix_p_3_avg'] = vix_p_3_avg
    df_final.loc[:,'vix_p_5_avg'] = vix_p_5_avg
    df_final.loc[:,'vix_p_10_avg'] = vix_p_10_avg
    df_final.loc[:,'vix_p_3_above'] = vix_p_3_above
    df_final.loc[:,'vix_p_5_above'] = vix_p_5_above
    df_final.loc[:,'vix_p_10_above'] = vix_p_10_above
    df_final.loc[:,'vix_p_3_below'] = vix_p_3_below
    df_final.loc[:,'vix_p_5_below'] = vix_p_5_below
    df_final.loc[:,'vix_p_10_below'] = vix_p_10_below
    df_final.loc[:,'fut_p_3_avg'] = fut_p_3_avg
    df_final.loc[:,'fut_p_5_avg'] = fut_p_5_avg
    df_final.loc[:,'fut_p_10_avg'] = fut_p_10_avg
    df_final.loc[:,'fut_p_3_above'] = fut_p_3_above
    df_final.loc[:,'fut_p_5_above'] = fut_p_5_above
    df_final.loc[:,'fut_p_10_above'] = fut_p_10_above
    df_final.loc[:,'fut_p_3_below'] = fut_p_3_below
    df_final.loc[:,'fut_p_5_below'] = fut_p_5_below
    df_final.loc[:,'fut_p_10_below'] = fut_p_10_below
    
    df_final.loc[:,'vix_p_3_3day_above'] = vix_p_3_above_3d
    df_final.loc[:,'vix_p_5_3day_above'] = vix_p_5_above_3d
    df_final.loc[:,'vix_p_10_3day_above'] = vix_p_10_above_3d
    df_final.loc[:,'vix_p_3_3day_below'] = vix_p_3_below_3d
    df_final.loc[:,'vix_p_5_3day_below'] = vix_p_5_below_3d
    df_final.loc[:,'vix_p_10_3day_below'] = vix_p_10_below_3d
    df_final.loc[:,'fut_p_3_3day_above'] = fut_p_3_above_3d
    df_final.loc[:,'fut_p_5_3day_above'] = fut_p_5_above_3d
    df_final.loc[:,'fut_p_10_3day_above'] = fut_p_10_above_3d
    df_final.loc[:,'fut_p_3_3day_below'] = fut_p_3_below_3d
    df_final.loc[:,'fut_p_5_3day_below'] = fut_p_5_below_3d
    df_final.loc[:,'fut_p_10_3day_below'] = fut_p_10_below_3d

    df_final.loc[:,'vix_p_3_exp_avg'] = vix_p_3_exp_avg
    df_final.loc[:,'vix_p_5_exp_avg'] = vix_p_5_exp_avg
    df_final.loc[:,'vix_p_10_exp_avg'] = vix_p_10_exp_avg
    df_final.loc[:,'vix_p_3_exp_above'] = vix_p_3_exp_above
    df_final.loc[:,'vix_p_5_exp_above'] = vix_p_5_exp_above
    df_final.loc[:,'vix_p_10_exp_above'] = vix_p_10_exp_above
    df_final.loc[:,'vix_p_3_exp_below'] = vix_p_3_exp_below
    df_final.loc[:,'vix_p_5_exp_below'] = vix_p_5_exp_below
    df_final.loc[:,'vix_p_10_exp_below'] = vix_p_10_exp_below
    df_final.loc[:,'fut_p_3_exp_avg'] = fut_p_3_exp_avg
    df_final.loc[:,'fut_p_5_exp_avg'] = fut_p_5_exp_avg
    df_final.loc[:,'fut_p_10_exp_avg'] = fut_p_10_exp_avg
    df_final.loc[:,'fut_p_3_exp_above'] = fut_p_3_exp_above
    df_final.loc[:,'fut_p_5_exp_above'] = fut_p_5_exp_above
    df_final.loc[:,'fut_p_10_exp_above'] = fut_p_10_exp_above
    df_final.loc[:,'fut_p_3_exp_below'] = fut_p_3_exp_below
    df_final.loc[:,'fut_p_5_exp_below'] = fut_p_5_exp_below
    df_final.loc[:,'fut_p_10_exp_below'] = fut_p_10_exp_below

    df_final.loc[:,'vix_BB_above'] = vix_bollinger_bands_above
    df_final.loc[:,'vix_BB_below'] = vix_bollinger_bands_below
    df_final.loc[:,'fut_BB_above'] = fut_bollinger_bands_above
    df_final.loc[:,'fut_BB_below'] = fut_bollinger_bands_below

    df_final.loc[:,'vix_disp3_avg'] = vix_disp_3
    df_final.loc[:,'vix_disp5_avg'] = vix_disp_5
    df_final.loc[:,'vix_disp3_above'] = vix_disp_3_above
    df_final.loc[:,'vix_disp5_above'] = vix_disp_5_above
    df_final.loc[:,'vix_disp3_below'] = vix_disp_3_below
    df_final.loc[:,'vix_disp5_below'] = vix_disp_5_below
    df_final.loc[:,'fut_disp3_avg'] = fut_disp_3
    df_final.loc[:,'fut_disp5_avg'] = fut_disp_5
    df_final.loc[:,'fut_disp3_above'] = fut_disp_3_above
    df_final.loc[:,'fut_disp5_above'] = fut_disp_5_above
    df_final.loc[:,'fut_disp3_below'] = fut_disp_3_below
    df_final.loc[:,'fut_disp5_below'] = fut_disp_5_below

    df_final.loc[:,'vix_momentum1'] = vix_mom1_5
    df_final.loc[:,'vix_momentum2'] = vix_mom2_5
    df_final.loc[:,'vix_roc'] = vix_roc
    df_final.loc[:,'vix_roc_above'] = vix_roc_above
    df_final.loc[:,'vix_roc_below'] = vix_roc_below
    df_final.loc[:,'fut_momentum1'] = fut_mom1_5
    df_final.loc[:,'fut_momentum2'] = fut_mom2_5
    df_final.loc[:,'fut_roc'] = fut_roc
    df_final.loc[:,'fut_roc_above'] = fut_roc_above
    df_final.loc[:,'fut_roc_below'] = fut_roc_below

    df_final.loc[:,'vix_william'] = vix_william
    df_final.loc[:,'vix_stochastic'] = vix_stochastic
    df_final.loc[:,'fut_william'] = fut_william
    df_final.loc[:,'fut_stochastic'] = fut_stochastic

    print('Phase 2 completed. Samples shown below:')
    print(df_final.tail(5))
    df_final.to_pickle(os.path.join(config.pickle_dir, 'vix_future_backup_phase_2.pkl'))
    




    ##vvix
    print('Phase 3 started ...')
    df_final = pd.read_pickle(os.path.join(config.pickle_dir, 'vix_future_backup_phase_2.pkl'))
    xls = pd.ExcelFile(config.vvix_table_path)
    df_vvix = pd.read_excel(xls, header=0)
    df_hl = df_vvix[['high date', 'VVIX high', 'VVIX low']]
    df_close = df_vvix[['close date', 'VVIX close']]    
    df_hl.loc[:,'vvix_hl'] = df_hl.apply(lambda x: x['VVIX high'] - x['VVIX low'], axis=1)
    # print(df_vvix.head(10))    
    vvix = []
    vvix_hl = []
    invalid_vvix_date = []
    for date in df_final['date'].tolist():
        if len(df_hl[df_hl['high date'] == date]['vvix_hl'].values) != 0 and len(df_close[df_close['close date'] == date]['VVIX close'].values) != 0:
            vvix_hl.append( df_hl[df_hl['high date'] == date]['vvix_hl'].values[0] )
            vvix.append( df_close[df_close['close date'] == date]['VVIX close'].values[0] )
        else:
            df_final = df_final.drop(df_final[df_final['date'] == date].index)
            invalid_vvix_date.append(date)
    df_final.loc[:,'vvix'] = vvix
    df_final.loc[:,'vvix_hl'] = vvix_hl
    df_final = df_final.reset_index(drop=True)
    print('invalid_date_due_to_vvix: {}'.format(invalid_vvix_date))




    
    #rsi
    xls = pd.ExcelFile(config.RSI_table_path)
    df_rsi = pd.read_excel(xls)
    # print(df_rsi)
    d3, d9, d14, d30 = [], [], [], []
    invalid_rsi_date = []
    for date in df_final['date'].tolist():
        try:
            d3.append( df_rsi[df_rsi['DATE'] == date]['RSI_3DAY'].values[0] )
            d9.append( df_rsi[df_rsi['DATE'] == date]['RSI_9DAY'].values[0] )
            d14.append( df_rsi[df_rsi['DATE'] == date]['RSI_14DAY'].values[0] )
            d30.append( df_rsi[df_rsi['DATE'] == date]['RSI_30DAY'].values[0] )
        except:
            invalid_rsi_date.append(date)
    df_final.loc[:,'RSI_3DAY'] = d3
    df_final.loc[:,'RSI_9DAY'] = d9
    df_final.loc[:,'RSI_14DAY'] = d14
    df_final.loc[:,'RSI_30DAY'] = d30
    print('invalid_date_due_to_RSI: {}'.format(invalid_rsi_date))
    print('Phase 3 completed. Samples shown below:')
    print(df_final.tail(5))
    save_pickle_path = os.path.join(config.pickle_dir, 'vix_{}_feature_unnorm.pkl'.format(df_final.tail(1)['date'].values[0]))
    df_final.to_pickle(save_pickle_path)
    print('Up to date feature (without normlisation and ground truth) generation competed.')





    ##normalize
    print('Phase 4 started ...')
    df_final = pd.read_pickle(os.path.join(config.pickle_dir, 'vix_future_backup_phase_3.pkl'))
    scaler = MinMaxScaler()
    for column in df_final.columns:
        if column != 'date' and column != 'm_1_close':
            df_final[column] = scaler.fit_transform(df_final[[column]])





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

    #df_final.drop(df_final.tail(5).index, inplace=True)
    #df_final = df_final.drop(columns=['m_1_close'])
    #df_final = df_final.reset_index(drop=True)
    print('Phase 4 completed. Samples shown below:')
    print(df_final.tail(10))





    ##output
    save_pickle_path = os.path.join(config.pickle_dir, 'vix_{}_feature_gt.pkl'.format(df_final.tail(1)['date'].values[0]))
    df_final.to_pickle(save_pickle_path)
    print('Up to date feature generation completed ! File saved as {}'.format(save_pickle_path))









def main_daily(config):    
    # global variables
    prior_day = config.prior_day
    _date_ = config.date
    print('current date: {}'.format(_date_))
    _year_ = _date_[:4]
    _month_ = _date_[4:6]
    _day_ = _date_[6:]

    pickle_path = os.path.join(config.pickle_dir,'vix_{}_feature_unnorm.pkl'.format(prior_day))
    print('loading past feature from {}'.format(pickle_path))
    df_final = pd.read_pickle(pickle_path)
    print('asserting if past feautre is up to date ... (if not, the script will terminate and you should change the configuration hyperparameter `feature_generate_mode` as `all`)')
    assert prior_day == df_final.tail(1)['date'].values[0]
    print('assertion completed. past features samples are shown below:')
    print(df_final.tail(5))



    # term structure
    mth_code = {'VXF':1, 'VXG':2, 'VXH':3, 'VXJ':4, 'VXK':5, 'VXM':6, 'VXN':7, 'VXQ':8, 'VXU':9, 'VXV':10, 'VXX':11, 'VXZ':12}
    subdir = _year_+'/'+_date_ #2011/20110101
    b_dates = pd.bdate_range(_year_+'/01/01', _year_+'/12/31')
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
        print('the input date is invalid') #TODO: auto terminate the script!
    
    pairs = list(itertools.combinations([8,7,6,5,4,3,2,1], 2))
    for pair in pairs:
        df_dic['m_{}-{}'.format(pair[0],pair[1])] = df_dic['m_{}_close'.format(pair[0])] - df_dic['m_{}_close'.format(pair[1])] 
    
    for i in range(2,9):
        del df_dic['m_{}_close'.format(i)]
        del df_dic['m_{}_high'.format(i)]
        del df_dic['m_{}_low'.format(i)]

    df_final = df_final.append(df_dic, ignore_index=True) 
    print('Phase 1 completed')




    # technical variables for 1mth vix future AND vix OR/AND vvix
    xls = pd.ExcelFile(config.vix_table_path)
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
    
    
    print('Phase 2 completed')



    ##vvix
    xls = pd.ExcelFile(config.vvix_table_path)
    df_vvix = pd.read_excel(xls, header=0)
    df_hl = df_vvix[['high date', 'VVIX high', 'VVIX low']]
    df_close = df_vvix[['close date', 'VVIX close']]
    
    vvix_hl = df_hl[df_hl['high date'] == _date_]['VVIX high'].values[0] - df_hl[df_hl['high date'] == _date_]['VVIX low'].values[0]
    vvix = df_close[df_close['close date'] == _date_]['VVIX close'].values[0]
    
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vvix')] = vvix
    df_final.iloc[new_data_idx, df_final.columns.get_loc('vvix_hl')] = vvix_hl





    #rsi
    xls = pd.ExcelFile(config.RSI_table_path)
    df_rsi = pd.read_excel(xls)

    d3 = df_rsi[df_rsi['DATE'] == _date_]['RSI_3DAY'].values[0] 
    d9 = df_rsi[df_rsi['DATE'] == _date_]['RSI_9DAY'].values[0] 
    d14 = df_rsi[df_rsi['DATE'] == _date_]['RSI_14DAY'].values[0] 
    d30 = df_rsi[df_rsi['DATE'] == _date_]['RSI_30DAY'].values[0] 

    df_final.iloc[new_data_idx, df_final.columns.get_loc('RSI_3DAY')] = d3
    df_final.iloc[new_data_idx, df_final.columns.get_loc('RSI_9DAY')] = d9
    df_final.iloc[new_data_idx, df_final.columns.get_loc('RSI_14DAY')] = d14
    df_final.iloc[new_data_idx, df_final.columns.get_loc('RSI_30DAY')] = d30

    print('daily feature generation (without normlisation and ground truth) compelted. Samples shown below:')
    print(df_final.tail(5))
    save_pickle_path = os.path.join(config.pickle_dir, 'vix_{}_feature_unnorm.pkl'.format(df_final.tail(1)['date'].values[0]))
    df_final.to_pickle(save_pickle_path)




    ##normalize
    scaler = MinMaxScaler()
    for column in df_final.columns:
        if column != 'date' and column != 'm_1_close':
            df_final[column] = scaler.fit_transform(df_final[[column]])
    




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





    ##output
    print('daily features & ground_truths updated !. Samples shown below:')
    print(df_final.tail(10))
    save_pickle_path = os.path.join(config.pickle_dir, 'vix_{}_feature_gt.pkl'.format(df_final.tail(1)['date'].values[0]))
    df_final.to_pickle(save_pickle_path)
    print('file saved as {}'.format(save_pickle_path))






if __name__ == "__main__":
    config = get_config()
    if config.feature_generate_mode == 'all':
        print('feature generation may take a while ...')
        main_backup(config)
    else:
        print('daily feature update started ...')
        main_daily(config)
