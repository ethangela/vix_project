# -*- coding: utf-8 -*-
from configs import get_config


def get_loader(pickle_path='Desktop/vix_future_preprocessed.pkl')
    all_data = pd.read_pickle('Desktop/vix_future_preprocessed.pkl')
    test_data_size = 30
    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]
    
    ###########################################
    
    return ground_truth, feature




class vanilla_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """Scoring LSTM"""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, full_size1)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(full_size1,full_size2)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(full_size2,1)
 
    def forward(self, features):
        """
        Args:
            features: [seq_len, batch_size, input_size=41] 
        Return:
            scores [batch_size, 1]
        """
        self.lstm.flatten_parameters()

        # [seq_len, batch_size, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)

        # [seq_len, batch_size, 1]
        features = nn.functional.relu(self.drop1(self.fc1(features)))
        features = nn.functional.relu(self.drop2(self.fc2(features)))
        predicts = self.fc3(features) 

        # [batch_size, seq_len]
        predicts = predicts.squeeze(-1)
        predicts = predicts.permute(1,0)

        return predicts


class Solver(object):
    def __init__(self, config=None, data_loader=None):
        self.config = config
        self.data_loader = data_loader


    # Build Modules
    def build(self):        
            self.model = vanilla_LSTM(
                self.config.lstm_input_size, 
                self.config.lstm_hidden_size,
                self.config.num_layers).cuda() #GPU setting

        if self.config.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
            self.model.train()
        
        elif self.config.mode == 'evaluate':
            self.model.eval()

    
    # Build Loss
    def MSE_loss(self):
        return nn.MSELoss()
    
    # Train 
    def train(self):
        step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch'):
            loss_history = []

            for batch_i, features_and_groundtruth in enumerate(self.data_loader): ####################
                
                # split features and groundtruth 
                features, ground_truth = features_and_groundtruth
                features = Variable(features).cuda() #GPU setting 
                ground_truth = Variable(ground_truth).cuda() #GPU setting 

                # train
                if self.config.verbose:
                    tqdm.write(str(epoch_i) + ': training for ' +str(batch_i))
        
                predicts = self.model(features) # [batch_size, seq_len]
                    
                
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

            # evaluate
            #self.evaluate(epoch_i)
            #self.model.train()


    def test(self, index, video_name):       
        # Load features
        features, ground_truth = features_and_groundtruth ####################
        features = Variable(features).cuda() #GPU setting
        
        # Load weights
        checkpoint = self.config.save_dir
        print(f'Load parameters from {checkpoint}')
        file = os.listdir(checkpoint)
        checkpoint_file_path = os.path.join(checkpoint, file[0])
        self.model.load_state_dict(torch.load(checkpoint_file_path))
        self.model.eval()
           
        # test 
        predicts = self.model(features) # [batch_size, seq_len]
        #predicts_ = predicts_.tolist()
        
        return predicts


if __name__ == '__main__':
    config = get_config(mode='train')
    video_loader, text_loader = get_loader(config.video_root_dir, config.mode, config.batch_size)
    solver = Solver(config, video_loader, text_loader)
    solver.build()
    solver.train()
