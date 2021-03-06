# -*- coding: utf-8 -*-
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import json
from tqdm import tqdm, trange
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import pandas as pd
import math
from configs import get_config


class VixData(Dataset):
    def __init__(self, feature_gt_pickle_path, feature_pickle_path, sequence_length, mode, result_mode):
        self.result_mode = result_mode
        all_data = pd.read_pickle(feature_gt_pickle_path) ###jun26
        data_length = len(all_data) - sequence_length + 1 
        self.train_length = data_length #int(data_length*(1-test_ratio)) ###jun26
        
        all_data_test = pd.read_pickle(feature_pickle_path) ###jun26

        def df_preprocess(dataframe, seq_len):
            date = dataframe.loc[:, 'date'].values #date = [ x.date() for x in dataframe.round(2).iloc[:].to_numpy()[:,0] ] #[length,]
            # feature = dataframe.round(2).iloc[:].to_numpy()[:,1:-2] #[length,input_size]
            # gt_3d = np.squeeze(dataframe.round(2).iloc[:].to_numpy()[:,-1]) #[length,]
            # gt_5d = np.squeeze(dataframe.round(2).iloc[:].to_numpy()[:,-2]) #[length,]
            feature = dataframe.round(2).loc[:, 'm_1_hl':'RSI_30DAY'].values ######################
            gt_1d = dataframe.round(2).loc[:, 'gt_1'].values ######################
            gt_2d = dataframe.round(2).loc[:, 'gt_2'].values
            gt_3d = dataframe.round(2).loc[:, 'gt_3'].values
            gt_4d = dataframe.round(2).loc[:, 'gt_4'].values 
            gt_5d = dataframe.round(2).loc[:, 'gt_5'].values 
            
            seq_features = [] 
            for seq_i in range(len(date)-seq_len+1):
                seq_feature = feature[seq_i:seq_len+seq_i]
                seq_features.append(seq_feature)
            seq_features = np.array(seq_features) #[length-seq_len+1, seq_len, input_size]
            seq_date = date[seq_len-1:] #[length-seq_len+1,]
            seq_gt_1d = gt_1d[seq_len-1:] #[length-seq_len+1,] ######################
            seq_gt_2d = gt_2d[seq_len-1:] #[length-seq_len+1,]
            seq_gt_3d = gt_3d[seq_len-1:] #[length-seq_len+1,]
            seq_gt_4d = gt_4d[seq_len-1:] #[length-seq_len+1,]
            seq_gt_5d = gt_5d[seq_len-1:] #[length-seq_len+1,]

            return seq_date, seq_features, seq_gt_1d, seq_gt_2d, seq_gt_3d, seq_gt_4d, seq_gt_5d ######################

        date, feature, gt_1d, gt_2d, gt_3d, gt_4d, gt_5d = df_preprocess(all_data, sequence_length) 

        def df_preprocess_test(dataframe, seq_len): ###jun26
            date = dataframe.loc[:, 'date'].values #date = [ x.date() for x in dataframe.round(2).iloc[:].to_numpy()[:,0] ] #[length,]
            # feature = dataframe.round(2).iloc[:].to_numpy()[:,1:-2] #[length,input_size]
            # gt_3d = np.squeeze(dataframe.round(2).iloc[:].to_numpy()[:,-1]) #[length,]
            # gt_5d = np.squeeze(dataframe.round(2).iloc[:].to_numpy()[:,-2]) #[length,]
            feature = dataframe.round(2).loc[:, 'm_1_hl':'RSI_30DAY'].values ######################
            
            seq_features = [] 
            for seq_i in range(len(date)-seq_len+1):
                seq_feature = feature[seq_i:seq_len+seq_i]
                seq_features.append(seq_feature)
            seq_features = np.array(seq_features) #[length-seq_len+1, seq_len, input_size]
            seq_date = date[seq_len-1:] #[length-seq_len+1,]

            return seq_date, seq_features

        test_date, test_feature, test_gt_1d, test_gt_2d, test_gt_3d, test_gt_4d, test_gt_5d = df_preprocess_test(all_data_test, sequence_length) ###jun26
        
        if mode.lower() == 'train':
            self.date, self.feature, self.gt_1d, self.gt_2d, self.gt_3d, self.gt_4d, self.gt_5d = date, feature, gt_1d, gt_2d, gt_3d, gt_4d, gt_5d ###jun26
        elif mode.lower() == 'evaluate':
            self.date, self.feature, self.gt_1d, self.gt_2d, self.gt_3d, self.gt_4d, self.gt_5d = date[-5:], feature[-5:], np.array([0]*5), np.array([0]*5), np.array([0]*5), np.array([0]*5), np.array([0]*5) ###jun26

        #print('features sample type {}, features sample {}, target sample {}'.format(type(list(self.feature[0])), self.feature[0], self.gt_3d[0]))
            
    def __len__(self):
        return len(self.date)

    def __getitem__(self, index):
        if self.result_mode == 1:
            gt = torch.Tensor([self.gt_1d[index]])
        elif self.result_mode == 2:
            gt = torch.Tensor([self.gt_2d[index]])
        elif self.result_mode == 3:
            gt = torch.Tensor([self.gt_3d[index]])
        elif self.result_mode == 4:
            gt = torch.Tensor([self.gt_4d[index]])
        elif self.result_mode == 5:
            gt = torch.Tensor([self.gt_5d[index]])
        
        feature = torch.Tensor(list(self.feature[index]))   

        date = self.date[index]
        
        return self.train_length, date, feature, gt

def get_loader(pickle_path, sequence_length, test_ratio, mode, result_mode, train_length_add, batch_size):
    return DataLoader(VixData(pickle_path, sequence_length, test_ratio, mode, result_mode, train_length_add), batch_size, shuffle=False)



class vanilla_LSTM(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, num_layers, full_hiden_size1, full_hiden_size2, drop_rate, out_length=1):
        """Scoring LSTM"""
        super().__init__()
        self.full_hiden_size2 = full_hiden_size2
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers, bidirectional=False)
        if self.full_hiden_size2 != 0:
            self.fc1 = nn.Linear(lstm_hidden_size, full_hiden_size1)
            self.drop1 = nn.Dropout(drop_rate)
            self.fc2 = nn.Linear(full_hiden_size1,full_hiden_size2)
            self.drop2 = nn.Dropout(drop_rate)
            self.fc3 = nn.Linear(full_hiden_size2,1) ###
            self.out_length = out_length
        else:
            self.fc1 = nn.Linear(lstm_hidden_size, full_hiden_size1)
            self.drop1 = nn.Dropout(drop_rate)
            self.fc3 = nn.Linear(full_hiden_size1,1) ###
            self.out_length = out_length
 
    def forward(self, features):
        """
        Args:
            features: [seq_len, batch_size, input_size=41] 
        Return:
            scores [batch_size, 1]
        """
        self.lstm.flatten_parameters()

        # lstm output
        outputs, (h_n, c_n) = self.lstm(features) #[seq_len, batch_size, hidden_size], ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size])

        # pick last certain step(s) of output 
        outputs = outputs[-1*self.out_length,:,:] #[out_length=1, batch_size, hidden_size]
        outputs = torch.squeeze(outputs) #[batch_size, hidden_size]

        # generate predict
        if self.full_hiden_size2 != 0:
            outputs = nn.functional.relu(self.drop1(self.fc1(outputs))) #[batch_size, full_size1]
            outputs = nn.functional.relu(self.drop2(self.fc2(outputs))) #[batch_size, full_size2] ###
            predicts = self.fc3(outputs) #[batch_size, 1]
        else:
            outputs = nn.functional.relu(self.drop1(self.fc1(outputs))) #[batch_size, full_size1]
            predicts = self.fc3(outputs) #[batch_size, 1]
        predicts = predicts.squeeze()#(-1) 
 
        return predicts #[batch_size,]



class Solver(object):
    def __init__(self, config=None, data_loader=None):
        self.config = config
        self.data_loader = data_loader


    # Build Modules
    def build(self):        
        self.model = vanilla_LSTM(
            self.config.input_size, 
            self.config.lstm_hidden_size,
            self.config.num_layers,
            self.config.full_hidden_size1,
            self.config.full_hidden_size2, 
            self.config.drop_rate).cuda() #GPU setting

        if self.config.mode.lower() == 'train':
            checkpoint_file_path = os.path.join(self.config.save_dir, self.config.ckpt_path)
            if os.path.isfile(checkpoint_file_path):
                # Load weights
                print(f'Load parameters from {checkpoint_file_path}')
                self.model.load_state_dict(torch.load(checkpoint_file_path))
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
            self.model.train()

        
        elif self.config.mode.lower() == 'evaluate':
            self.model.eval()

    
    # Build Loss
    def MSE_loss(self):
        return nn.MSELoss() 
        # return nn.L1Loss()
    

    # Train 
    def train(self):
        step = 0
        for epoch_i in trange(self.config.start_epoch,self.config.start_epoch+self.config.n_epochs, desc='Epoch'):
            loss_history = []    
            date_list, out_list, target_list = [], [], []
            for batch_i, features_and_groundtruth in enumerate(self.data_loader): 
                
                # split features and groundtruth 
                train_len, date, features, ground_truth = features_and_groundtruth #[batch_size,] [batch_size, seq_len, input_size], [batch_size,]
                # print('BatchIdx {}, features.shape {}, target.shape {}'.format(batch_i, features.shape, ground_truth.shape))
                features = features.permute(1,0,2) #[seq_len, batch_size, input_size]
                features = Variable(features).cuda() #GPU setting 
                ground_truth = Variable(ground_truth.squeeze()).cuda() #GPU setting 

                # train
                # if self.config.verbose:
                #     tqdm.write(str(epoch_i) + ': training for batch' +str(batch_i))
        
                predicts = self.model(features.detach()) # [batch_size,]
                #date_list.append(date) 
                #out_list.append(predicts.item())
                #target_list.append(ground_truth.item())
                
                # loss calculation 
                #if epoch_i%10 == 0:
                #    print('SHAPE CHECK. output: {} with shape {}'.format(predicts, predicts.size()))
                #    print('SHAPE CHECK. target: {} with shape {}'.format(ground_truth, ground_truth.size()))
                entropy_loss_function = self.MSE_loss() 
                entropy_loss = entropy_loss_function(predicts, ground_truth)
                #tqdm.write(f'entropy loss {entropy_loss.item():.3f}')
                
                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                # backward propagation 
                entropy_loss.backward()
                
                # Gradient cliping
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                
                # parameters update 
                self.optimizer.step()
                
                # batch loss record 
                loss_history.append(entropy_loss.data)               
                
                # tesorboard plotting 
                # if entropy_loss > 5: 
                #    print('entropy_loss at date {} at step {} is {}'.format(date, step, entropy_loss.data))

                step += 1
            
            # average loss per epoch  
            e_loss = torch.stack(loss_history).mean()
            #print('date', date_list)
            #print('predict', out_list)
            #print('target', target_list)

            # tesorboard plotting 
            # if self.config.verbose:
            #     tqdm.write('Plotting...')
            print('avg_epoch_loss at epoch {} is {}'.format(epoch_i, e_loss))

            # Save parameters at checkpoint
            if os.path.isdir(self.config.save_dir) is False:
                os.mkdir(self.config.save_dir)
            #test_ratio_inf = ''.join(str(self.config.test_ratio).split('.'))
            if len(train_len) > 1:
                train_len = int(train_len[0])
            if epoch_i%50 == 0:
                checkpint_name = 'date_{}_seq_len_{}_num_lay_{}_hid1_{}_ful1_{}_ful2_{}_train_len_{}_epoch_{}_gt{}.pkl'.format(self.config.date, self.config.sequence_length, 
                self.config.num_layers, self.config.lstm_hidden_size, self.config.full_hidden_size1, self.config.full_hidden_size2, train_len, epoch_i, str(self.config.result_mode))
                ckpt_path = os.path.join(self.config.save_dir, checkpint_name)
                tqdm.write(f'Save parameters at {ckpt_path}')
                torch.save(self.model.state_dict(), ckpt_path)


    def evaluate(self):   
        # Load weights
        print(f'Load parameters from {self.config.ckpt_path}')
        checkpoint_file_path = os.path.join(self.config.save_dir, self.config.ckpt_path)
        self.model.load_state_dict(torch.load(checkpoint_file_path))
        self.model.eval()

        # Load data and output the result
        loss_history = []
        date_list, out_list, target_list = [], [], []
        for batch_i, features_and_groundtruth in enumerate(self.data_loader):                
            # split features and groundtruth 
            train_length, date, features, ground_truth = features_and_groundtruth #[batch_size,] [batch_size, seq_len, input_size], [batch_size,]
            features = features.permute(1,0,2) #[seq_len, batch_size, input_size]
            features = Variable(features).cuda() #GPU setting 
            # ground_truth = Variable(ground_truth).cuda() #GPU setting 

            # output 
            predicts = self.model(features.detach()) # [batch_size,]
            out_list.append(predicts.item())
            date_list.append(date[0]) 

        # print
        print('predict', out_list)
        print('date', date_list)

        print('important out', out_list[-1]) ###jun26
        print('important date', date_list[-1]) ###jun26

        ###############need out log

if __name__ == '__main__':
    config = get_config()
    vix_loader = get_loader(config.pickle_path, config.sequence_length, config.test_ratio, config.mode, config.result_mode, config.train_length_add, config.batch_size)
    solver = Solver(config, vix_loader)
    solver.build()
    if config.mode == 'train':
        solver.train()
    else:
        solver.evaluate()