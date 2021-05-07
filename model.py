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
    def __init__(self, pickle_path, sequence_length, test_ratio, mode, result_mode):
        self.result_mode = result_mode
        all_data = pd.read_pickle(pickle_path)
        data_length = len(all_data) - sequence_length + 1
        train_length = int(data_length*(1-test_ratio))

        def df_preprocess(dataframe, seq_len):
            date = dataframe.loc[:, 'date'].values #date = [ x.date() for x in dataframe.round(2).iloc[:].to_numpy()[:,0] ] #[length,]
            # feature = dataframe.round(2).iloc[:].to_numpy()[:,1:-2] #[length,input_size]
            # gt_3d = np.squeeze(dataframe.round(2).iloc[:].to_numpy()[:,-1]) #[length,]
            # gt_5d = np.squeeze(dataframe.round(2).iloc[:].to_numpy()[:,-2]) #[length,]
            feature = dataframe.round(2).loc[:, 'm_1_hl':'vvix_hl'].values ######################
            gt_1d = dataframe.round(2).loc[:, 'gt_1'].values ######################
            gt_3d = dataframe.round(2).loc[:, 'gt_3'].values 
            gt_5d = dataframe.round(2).loc[:, 'gt_5'].values 
            
            seq_features = [] 
            for seq_i in range(len(date)-seq_len+1):
                seq_feature = feature[seq_i:seq_len+seq_i]
                seq_features.append(seq_feature)
            seq_features = np.array(seq_features) #[length-seq_len+1, seq_len, input_size]
            seq_date = date[seq_len-1:] #[length-seq_len+1,]
            seq_gt_1d = gt_1d[seq_len-1:] #[length-seq_len+1,] ######################
            seq_gt_3d = gt_3d[seq_len-1:] #[length-seq_len+1,]
            seq_gt_5d = gt_5d[seq_len-1:] #[length-seq_len+1,]

            return seq_date, seq_features, seq_gt_1d, seq_gt_3d, seq_gt_5d ######################

        date, feature, gt_1d, gt_3d, gt_5d = df_preprocess(all_data, sequence_length) ######################

        if mode.lower() == 'train':
            self.date, self.feature, self.gt_1d, self.gt_3d, self.gt_5d = date[:train_length], feature[:train_length], gt_1d[:train_length], gt_3d[:train_length], gt_5d[:train_length] ######################
        elif mode.lower() == 'evaluate':
            self.date, self.feature, self.gt_1d, self.gt_3d, self.gt_5d = date[train_length:], feature[train_length:], gt_1d[train_length:], gt_3d[train_length:], gt_5d[train_length:] ######################

        #print('features sample type {}, features sample {}, target sample {}'.format(type(list(self.feature[0])), self.feature[0], self.gt_3d[0]))
            
    def __len__(self):
        return len(self.date)

    def __getitem__(self, index):
        if self.result_mode == 1:
            gt = torch.Tensor([self.gt_1d[index]])
        elif self.result_mode == 3:
            gt = torch.Tensor([self.gt_3d[index]])
        elif self.result_mode == 5:
            gt = torch.Tensor([self.gt_5d[index]])
        
        feature = torch.Tensor(list(self.feature[index]))   

        date = self.date[index]
        
        return date, feature, gt

def get_loader(pickle_path, sequence_length, test_ratio, mode, result_mode, batch_size):
    return DataLoader(VixData(pickle_path, sequence_length, test_ratio, mode, result_mode), batch_size, shuffle=False)



class vanilla_LSTM(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, num_layers, full_hiden_size1, full_hiden_size2, drop_rate, out_length=1):
        """Scoring LSTM"""
        super().__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers, bidirectional=False)
        self.fc1 = nn.Linear(lstm_hidden_size, full_hiden_size1)
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(full_hiden_size1,full_hiden_size2)
        self.drop2 = nn.Dropout(drop_rate)
        self.fc3 = nn.Linear(full_hiden_size2,1) ###
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
        outputs = nn.functional.relu(self.drop1(self.fc1(outputs))) #[batch_size, full_size1]
        outputs = nn.functional.relu(self.drop2(self.fc2(outputs))) #[batch_size, full_size2] ###
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
            if epoch_i%100 == 0:
                checkpint_name = 'seq_{}_epoch_{}_gt{}.pkl'.format(self.config.sequence_length, epoch_i, str(self.config.result_mode))
                ckpt_path = os.path.join(self.config.save_dir, checkpint_name)
                tqdm.write(f'Save parameters at {ckpt_path}')
                torch.save(self.model.state_dict(), ckpt_path)


    def evaluate(self):   
        # Load weights
        print(f'Load parameters from {self.config.save_dir}')
        checkpoint_file_path = os.path.join(self.config.save_dir, self.config.ckpt_path)
        self.model.load_state_dict(torch.load(checkpoint_file_path))
        self.model.eval()

        # Load data and output the result
        loss_history = []
        date_list, out_list, target_list = [], [], []
        for batch_i, features_and_groundtruth in enumerate(self.data_loader):                
            # split features and groundtruth 
            date, features, ground_truth = features_and_groundtruth #[batch_size,] [batch_size, seq_len, input_size], [batch_size,]
            features = features.permute(1,0,2) #[seq_len, batch_size, input_size]
            features = Variable(features).cuda() #GPU setting 
            ground_truth = Variable(ground_truth).cuda() #GPU setting 

            # output 
            predicts = self.model(features.detach()) # [batch_size,]
            #out_list.append(predicts.item())
            #date_list.append(date) 
            #target_list.append(ground_truth.item())

            # loss calculation 
            entropy_loss_function = self.MSE_loss() 
            entropy_loss = entropy_loss_function(predicts, ground_truth)
            tqdm.write(f'date {date}, entropy loss {entropy_loss.item():.3f}')

            # batch loss record 
            loss_history.append(entropy_loss.data)  

        # average loss per epoch  
        e_loss = torch.stack(loss_history).mean()

        # tesorboard plotting 
        print('avg_evaluation_loss is {}'.format(e_loss))
        #print('predict', out_list)
        #print('target', target_list)
        #print('date', date_list)

if __name__ == '__main__':
    config = get_config()
    vix_loader = get_loader(config.pickle_path, config.sequence_length, config.test_ratio, config.mode, config.result_mode, config.batch_size)
    solver = Solver(config, vix_loader)
    solver.build()
    if config.mode == 'train':
        solver.train()
    else:
        solver.evaluate()
