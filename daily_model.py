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
from daily_configs import get_config
import pickle


class VixData(Dataset):
    def __init__(self, pickle_dir, train_pickle_path, sequence_length, train_mode, result_mode):
        self.train_mode = train_mode

        def df_preprocess(pck_dir, pck_path, seq_len, gt_mode):
            gt_dic = {1:'gt_1', 2:'gt_2', 3:'gt_3', 4:'gt_4', 5:'gt_5'}
            pickle_path = os.path.join(pck_dir, pck_path)
            all_data = pd.read_pickle(pickle_path)
            # all_data.drop(all_data.tail(gt_mode).index, inplace=True)
            # all_data = all_data.reset_index(drop=True)
            
            train_length = len(all_data) - seq_len + 1
            
            feature = all_data.loc[:, 'm_1_hl':'RSI_30DAY'].values #changeable if more future features are available
            seq_features = [] 
            for seq_i in range(train_length):
                seq_feature = feature[seq_i:seq_len+seq_i]
                seq_features.append(seq_feature)
            seq_features = np.array(seq_features) #[length-seq_len+1, seq_len, input_size]
            
            date = all_data.loc[:, 'date'].values 
            seq_date = date[seq_len-1:] #[length-seq_len+1,]
            
            ground_truth = all_data.loc[:, gt_dic[gt_mode]].values 
            seq_gt = ground_truth[seq_len-1:] #[length-seq_len+1,] 

            return seq_date, seq_features, seq_gt 

        date, feature, gt = df_preprocess(pickle_dir, train_pickle_path, sequence_length, result_mode) 
        
        if train_mode.lower() == 'train':
            self.date, self.feature, self.gt = date[:-result_mode], feature[:-result_mode], gt[:-result_mode]
        elif train_mode.lower() == 'test':
            self.date, self.feature = [date[-1]], np.expand_dims(feature[-1,:,:], axis=0)

            
    def __len__(self):
        return len(self.date)

    def __getitem__(self, index):
        feature = torch.Tensor(list(self.feature[index]))   
        date = self.date[index]
        if self.train_mode.lower() == 'train':
            gt = torch.Tensor([self.gt[index]])
            return date, feature, gt
        elif self.train_mode.lower() == 'test':
            return date, feature

def get_loader(pickle_dir, train_pickle_path, sequence_length, train_mode, result_mode, batch_size_):
    if train_mode == 'test':
        batch_size = 1
    else:
        batch_size = batch_size_
    return DataLoader(VixData(pickle_dir, train_pickle_path, sequence_length, train_mode, result_mode), batch_size, shuffle=False)



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
            # checkpoint_file_path = os.path.join(self.config.save_dir, self.config.ckpt_path) #depreciated on Jul,4,2021 by Young
            # if os.path.isfile(checkpoint_file_path):
            #     # Load weights
            #     print(f'Load parameters from {checkpoint_file_path}')
            #     self.model.load_state_dict(torch.load(checkpoint_file_path))
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
            self.model.train()

        elif self.config.mode.lower() == 'test':
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
                date, features, ground_truth = features_and_groundtruth #[batch_size,] [batch_size, seq_len, input_size], [batch_size,]
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                
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
            if epoch_i%50 == 0:
                checkpint_name = 'date_{}_gt_{}_epoch_{}.pkl'.format(self.config.date, str(self.config.result_mode), epoch_i)
                ckpt_path = os.path.join(self.config.save_dir, checkpint_name)
                tqdm.write(f'Saving parameters at {ckpt_path}')
                torch.save(self.model.state_dict(), ckpt_path)


    def evaluate(self):   
        # Load weights
        print(f'Load parameters from {self.config.ckpt_path}')
        checkpoint_file_path = os.path.join(self.config.save_dir, self.config.ckpt_path)
        self.model.load_state_dict(torch.load(checkpoint_file_path))
        self.model.eval()

        # Load data and output the result
        date_list, out_list = [], []
        for batch_i, dates_features in enumerate(self.data_loader):                
            # split features and groundtruth 
            date, features = dates_features #[batch_size,] [batch_size, seq_len, input_size]
            features = features.permute(1,0,2) #[seq_len, batch_size, input_size]
            features = Variable(features).cuda() #GPU setting 

            # output 
            predicts = self.model(features.detach()) # [batch_size,]
            out_list.append(predicts.item())
            date_list.append(date[0]) 

        assert len(date_list) == len(out_list) == 1
        save_pickle_path = os.path.join(self.config.pickle_dir, date_list[0]+'_forward_'+str(self.config.result_mode)+'.pkl')
        with open(save_pickle_path, 'wb') as f:
            pickle.dump(out_list, f)

if __name__ == '__main__':
    config = get_config()
    print(config.date)
    for i in range(1,6):
        config.result_mode = i
        config.ckpt_path = 'date_{}_gt_{}_epoch_{}.pkl'.format(config.date, str(config.result_mode), 200)
        config.train_pickle_path = 'vix_{}_feature_gt.pkl'.format(config.date)
        vix_loader = get_loader(config.pickle_dir, config.train_pickle_path, config.sequence_length, config.mode, config.result_mode, config.batch_size)
        solver = Solver(config, vix_loader)
        solver.build()
        if config.mode == 'train':
            solver.train()
        else:
            solver.evaluate()



