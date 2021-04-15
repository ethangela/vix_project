# -*- coding: utf-8 -*-
import torch
import torch.nn as nn 
import math
from configs import get_config
from torch.utils.data import Dataset, DataLoader



class VixData(Dataset):
    def __init__(self, pickle_path, sequence_length, test_ratio, mode, result_mode):
        self.result_mode = result_mode
        all_data = pd.read_pickle(pickle_path)
        data_length = len(all_data)
        train_length = int(data_length*(1-test_ratio)))

        def df_preprocess(dataframe, seq_len):
            date = [ x.date() for x in split_df.round(2).iloc[:].to_numpy()[:,0] ] #[length,]
            feature = split_df.round(2).iloc[:].to_numpy()[:,1:-2] #[length,input_size]
            gt_3d = np.squeeze(split_df.round(2).iloc[:].to_numpy()[:,-1]) #[length,]
            gt_5d = np.squeeze(split_df.round(2).iloc[:].to_numpy()[:,-2]) #[length,]
            
            seq_features = [] 
            for seq_i in range(len(date)-seq_len+1):
                seq_feature = feature[seq_i:seq_len+seq_i]
                seq_features.append(seq_feature)
            seq_features = np.array(seq_features) #[length-seq_len+1, seq_len, input_size]
            seq_date = date[seq_len-1:] #[length-seq_len+1,]
            seq_gt_3d = gt_3d[seq_len-1:] #[length-seq_len+1,]
            seq_gt_5d = gt_5d[seq_len-1:] #[length-seq_len+1,]

            return seq_features, seq_date, seq_gt_3d, seq_gt_5d

        date, feature, gt_3d, gt_5d = df_preprocess(all_data, sequence_length)

        if mode.lower() == 'train':
            self.date, self.feature, self.gt_3d, self.gt_5d = date[:train_length], feature[:train_length], gt_3d[:train_length], gt_5d[:train_length]
        elif mode.lower() == 'evaluate':
            self.date, self.feature, self.gt_3d, self.gt_5d = date[train_length:], feature[train_length:], gt_3d[train_length:], gt_5d[train_length:]
            
    def __len__(self):
        return len(self.date)

    def __getitem__(self, index):
        if self.result_mode == 3:
            gt = torch.Tensor(self.gt_3d[index])
        elif self.result_mode == 5:
            gt = torch.Tensor(self.gt_5d[index])
        
        feature = torch.Tensor(self.feature[index])   

        date = self.date[index]
        
        return date, feature, gt

def get_loader(pickle_path, test_ratio, mode, result_mode, batch_size):
    return DataLoader(VixData(pickle_path, test_ratio, mode, result_mode), batch_size)



class vanilla_LSTM(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, num_layers, full_hiden_size1, full_hiden_size2, drop_rate, out_length=1):
        """Scoring LSTM"""
        super().__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers, bidirectional=False)
        self.fc1 = nn.Linear(lstm_hidden_size, full_hiden_size1)
        self.drop1 = nn.Dropout(drop_rate)
        # self.fc2 = nn.Linear(full_hiden_size1,full_hiden_size2)
        # self.drop2 = nn.Dropout(drop_rate)
        # self.fc3 = nn.Linear(full_hiden_size2,1)
        self.fc3 = nn.Linear(full_hiden_size1,1)
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
        # outputs = nn.functional.relu(self.drop2(self.fc2(outputs))) #[batch_size, full_size2]
        predicts = self.fc3(outputs) #[batch_size, 1]
        predicts = predicts.squeeze(-1) 
 
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
                self.config.full_hiden_size1,
                self.config.full_hiden_size2, 
                self.config.drop_rate).cuda() #GPU setting

        if self.config.mode.lower() == 'train':
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
        for epoch_i in trange(self.config.n_epochs, desc='Epoch'):
            loss_history = []

            for batch_i, features_and_groundtruth in enumerate(self.data_loader): 
                
                # split features and groundtruth 
                date, features, ground_truth = features_and_groundtruth #[batch_size,] [batch_size, seq_len, input_size], [batch_size,]
                features = features.permute(1,0,2) #[seq_len, batch_size, input_size]
                features = Variable(features).cuda() #GPU setting 
                ground_truth = Variable(ground_truth).cuda() #GPU setting 

                # train
                if self.config.verbose:
                    tqdm.write(str(epoch_i) + ': training for ' +str(batch_i))
        
                predicts = self.model(features.detach()) # [batch_size,]
                    
                
                # loss calculation 
                entropy_loss_function = self.MSE_loss() 
                entropy_loss = entropy_loss_function(predicts, ground_truth)
                tqdm.write(f'entropy loss {entropy_loss.item():.3f}')
                
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
                if self.config.verbose:
                    tqdm.write('Plotting...')
                print('entropy_loss at step {} is {}'.format(step, entropy_loss.data))

                step += 1
            
            # average loss per epoch  
            e_loss = torch.stack(loss_history).mean()

            # tesorboard plotting 
            if self.config.verbose:
                tqdm.write('Plotting...')
            print('avg_epoch_loss at epoch {} is {}'.format(epoch_i, e_loss))

            # Save parameters at checkpoint
            if os.path.isdir(self.config.save_dir) is False:
                os.mkdir(self.config.save_dir)
            ckpt_path = str(self.config.save_dir) + f'_epoch-{epoch_i}.pkl'
            tqdm.write(f'Save parameters at {ckpt_path}')
            torch.save(self.model.state_dict(), ckpt_path)


    def evaluate(self):   
        # Load weights
        checkpoint = self.config.save_dir
        print(f'Load parameters from {checkpoint}')
        file = os.listdir(checkpoint)
        checkpoint_file_path = os.path.join(checkpoint, file[0])
        self.model.load_state_dict(torch.load(checkpoint_file_path))
        self.model.eval()

        # Load data and output the result
        loss_history = []
        for batch_i, features_and_groundtruth in enumerate(self.data_loader):                
            # split features and groundtruth 
            date, features, ground_truth = features_and_groundtruth #[batch_size,] [batch_size, seq_len, input_size], [batch_size,]
            features = features.permute(1,0,2) #[seq_len, batch_size, input_size]
            features = Variable(features).cuda() #GPU setting 
            ground_truth = Variable(ground_truth).cuda() #GPU setting 

            # output 
            predicts = self.model(features.detach()) # [batch_size,]
            
            # loss calculation 
            entropy_loss_function = self.MSE_loss() 
            entropy_loss = entropy_loss_function(predicts, ground_truth)
            tqdm.write(f'entropy loss {entropy_loss.item():.3f}')

            # batch loss record 
            loss_history.append(entropy_loss.data)  

        # average loss per epoch  
        e_loss = torch.stack(loss_history).mean()

        # tesorboard plotting 
        if self.config.verbose:
            tqdm.write('Plotting...')
        print('avg_evaluation_loss is {}'.format(e_loss))


if __name__ == '__main__':