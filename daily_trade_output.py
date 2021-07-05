import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import pickle
from daily_configs import get_config
import sys


def trade_insight_daily(config):    
    _date_ = config.date

    #target
    targets = []
    feature_pickle_path = os.path.join(config.pickle_dir, 'vix_{}_feature_gt.pkl'.format(_date_))
    all_data = pd.read_pickle(feature_pickle_path)
    date_idx = all_data[all_data['date'] == _date_].index[0]
    for j in range(date_idx-5+1,date_idx+1):
        target = all_data.iloc[j]['m_1_close']
        targets.append(target)
    

    #predict
    predicts = []
    for i in range(1,6):
        load_pickle_path = os.path.join(config.pickle_dir, str(_date_)+'_forward_'+str(i)+'.pkl')
        with open(load_pickle_path, 'rb') as f:
            predict = pickle.load(f)
        predicts.append(predict[0])

    
    #trade
    def entry_point_minmax(pre, tar, ratio):
        if ( max(pre) - tar[-1] ) / tar[-1] > ratio and ( max(tar) - tar[-1] ) / tar[-1] > ratio :
            return 1
        else:
            return 0
            
    def exit_point_minmax(pre, tar, ratio):
        if ( tar[-1] - min(pre) ) / tar[-1] > ratio and ( tar[-1] - min(tar) ) / tar[-1] > ratio :
            return 1
        else:
            return 0
    
    def entry_point_mean(pre, tar, ratio):
        if ( np.mean(pre) - tar[-1] ) / tar[-1] > ratio and ( np.mean(tar) - tar[-1] ) / tar[-1] > ratio :
            return 1
        else:
            return 0

    def exit_point_mean(pre, tar, ratio):
        if ( tar[-1] - np.mean(pre) ) / tar[-1] > ratio and ( tar[-1] - np.mean(tar) ) / tar[-1] > ratio :
            return 1
        else:
            return 0

    def entry_point_3consecutive(pre, tar):
        surge = tar[-1] < pre[0] and pre[0] < pre[1] and pre[1] < pre[2]
        plunge = tar[-4] > tar[-3] and tar[-3] > tar[-2] and tar[-2] > tar[-1]
        if plunge and surge:
            return 1
        else:
            return 0

    def exit_point_3consecutive(pre, tar):
        plunge = tar[-1] > pre[0] and pre[0] > pre[1] and pre[1] > pre[2]
        surge = tar[-4] < tar[-3] and tar[-3] < tar[-2] and tar[-2] < tar[-1]
        if plunge and surge:
            return 1
        else:
            return 0

    def entry_point_2consecutive(pre, tar):
        surge = tar[-1] < pre[0] and pre[0] < pre[1] 
        plunge = tar[-3] > tar[-2] and tar[-2] > tar[-1]
        if plunge and surge:
            return 1
        else:
            return 0

    def exit_point_2consecutive(pre, tar):
        plunge = tar[-1] > pre[0] and pre[0] > pre[1]
        surge = tar[-3] < tar[-2] and tar[-2] < tar[-1]
        if plunge and surge:
            return 1
        else:
            return 0

    def entry_point_1consecutive_ratio(pre, tar, ratio):
        surge = tar[-1] < pre[0] and (pre[0] - tar[-1]) / tar[-1] > ratio
        plunge = tar[-2] > tar[-1] and (tar[-2] - tar[-1]) / tar[-1] > ratio
        if plunge and surge:
            return 1
        else:
            return 0

    def exit_point_1consecutive_ratio(pre, tar, ratio):
        plunge = tar[-1] > pre[0] and (tar[-1] - pre[0]) / tar[-1] > ratio
        surge = tar[-2] < tar[-1] and (tar[-1] - tar[-2]) / tar[-1] > ratio
        if plunge and surge:
            return 1
        else:
            return 0

    entry_minmax = entry_point_minmax(predicts, targets, 0.07)
    exit_minmax = exit_point_minmax(predicts, targets, 0.12)
    
    entry_mean = entry_point_mean(predicts, targets, 0.08)
    exit_mean = exit_point_mean(predicts, targets, 0.08)

    entry_3con = entry_point_3consecutive(predicts, targets)
    exit_3con = exit_point_3consecutive(predicts, targets)

    entry_2con = entry_point_2consecutive(predicts, targets)
    exit_2con = exit_point_2consecutive(predicts, targets)

    entry_1con = entry_point_1consecutive_ratio(predicts, targets, 0.03)
    exit_1con = exit_point_1consecutive_ratio(predicts, targets, 0.04)

    entry_join = np.mean(entry_minmax + entry_mean + entry_3con + entry_2con + entry_1con)
    exit_join = np.mean(exit_minmax + exit_mean + exit_3con + exit_2con + exit_1con)

    buy, sell = 'No Action', 'No Action'
    
    if entry_join == 3.:
        buy = 'b'
    elif entry_join >= 4.:
        buy = 'b+'
    
    if exit_join == 3.:
        sell = 's'
    elif exit_join >= 4.:
        sell = 's+'

    log_date = 'date: {}'.format(_date_) 
    log_truth = 'close price at today: {}, 1 day backward: {}, 2 days backward: {}, 3 days backward: {}, 4 days backward: {}'.format(targets[-1], targets[-2], 
        targets[-3], targets[-4], targets[-5])
    log_predict = 'predicted price in 1 day forward: {}, 2 days forward: {}, 3 days forward: {}, 4 days forward: {}, 5 days forward: {}, '.format(predicts[0], 
        predicts[1], predicts[2], predicts[3], predicts[4])
    log_buy_sell = 'Buy signal: {}, Sell signal: {}'.format(buy, sell)

    print(log_date)
    print(log_truth)
    print(log_predict)
    print(log_buy_sell)
    

if __name__ == '__main__':    
    config = get_config()
    file = sys.stdout   
    sys.stdout = open(config.output_file_path, 'w')   
    trade_insight_daily(config)