import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict, Counter
from itertools import combinations
import json
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
import optuna
import pickle
import sys
### import Dataset prepartion and model training classes from Auxiliary_Codes folder
from BS_LS_DataSet import BS_LS_DataSet_Prep, BS_LS_upper_lower_concat_rcm


### Model 1 input sequence 4 X 400 with 2CNN layer
### retrain the base model 1 with selected hyperparameters
### 10000 training size used here for example
class Model1_optuna_10000(nn.Module):
    '''
        This model take in input sequence 4 X 400 with 1 CNN layer
    '''

    def __init__(self, trial):
        super(Model1_optuna_10000, self).__init__()
        ### first CNN layer
        #         self.out_channel1 = trial.suggest_categorical('out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 17
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 400 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21
        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('maxpool2', [5, 10, 20])
        self.maxpool2 = 10
        self.conv2_out_dim = 400 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class ConcatModel1_optuna_10000(nn.Module):
    def __init__(self, trial):
        super(ConcatModel1_optuna_10000, self).__init__()

        ### cnn for the concatenated sequence
        self.cnn = Model1_optuna_10000(trial)

        # this is for two convlayer
        self.out_dim = self.cnn.conv2_out_dim
        self.out_channel = self.cnn.out_channel2

        self.fc1_input_dim = self.out_channel * self.out_dim

        #         self.fc1_out = trial.suggest_categorical('concat_fc1_out', [128, 256, 512])
        self.fc1_out = 128
        self.fc1 = nn.Linear(self.fc1_input_dim, self.fc1_out)

        self.fc1_bn = nn.BatchNorm1d(self.fc1_out)

        #         dropout_rate_fc1 = trial.suggest_categorical("concat_dropout_rate_fc1",  [0, 0.1, 0.2, 0.4])
        dropout_rate_fc1 = 0
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc layer2
        # use dimension output with nn.CrossEntropyLoss()
        #         self.fc2_out = trial.suggest_categorical('concat_fc2_out', [4, 8, 16, 32])
        self.fc2_out = 4
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

        self.fc2_bn = nn.BatchNorm1d(self.fc2_out)

        #         dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2",[0, 0.1, 0.2, 0.4])
        dropout_rate_fc2 = 0

        self.drop_nn2 = nn.Dropout(p=dropout_rate_fc2)

        self.fc3 = nn.Linear(self.fc2_out, 2)

    def forward(self, seq_upper_lower_feature):
        x = self.cnn(seq_upper_lower_feature)
        # feed the concatenated feature to fc1
        out = self.fc1(x)
        out = self.drop_nn1(torch.relu(self.fc1_bn(out)))

        out = self.fc2(out)
        out = self.drop_nn2(torch.relu(self.fc2_bn(out)))

        out = self.fc3(out)
        return out


def retraining(model, dataset, model_folder):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_size = 32

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #     print(len(train_loader))

    criterion = nn.CrossEntropyLoss()

    model = model('trial').to(device=device)
    #     print(model)

    optimizer_name = 'Adam'
    lr = 0.00016663145779864379
    l2_lambda = 4.828346469350033e-07
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=l2_lambda)

    epochs = 150  ### reduce the epochs from 150 to 100 to reduce the potential overfitting

    for epoch in range(epochs):
        #         print(f"I'am in the epoch {epoch}")
        model.train()
        # record the training loss
        running_loss = 0.0

        ## deal with different number of features in different dataset with star* notation
        for *features, train_labels in train_loader:
            ### this line is just for nn.CrossEntropy loss otherwise can be safely removed
            train_labels = train_labels.type(torch.LongTensor)

            train_labels = train_labels.to(device)
            features = [i.to(device) for i in features]

            # forward pass
            train_preds = model(*features)
            loss = criterion(train_preds, train_labels)
            # backward pass
            optimizer.zero_grad()  # empty the gradient from last round

            # calculate the gradient
            loss.backward()
            # update the parameters
            optimizer.step()
            running_loss += loss.item()

        #         print(f"I'am finished the epoch {epoch} training")
        ## start model validation

        model.eval()
        with torch.no_grad():
            # first evaluate the training acc ## don't need to evaluate the training acc
            correct, total = 0.0, 0.0
            for *features, train_labels in train_loader:
                ### this type conversion is just used for nn.CrossEntropy loss
                ### otherwise can be safely removed
                train_labels = train_labels.to(device)
                features = [i.to(device) for i in features]

                # get the predition with the model parameters updated after each epoch
                preds = model(*features)
                # prediction for the nn.CrossEntropy loss
                _, preds_labels = torch.max(preds, 1)
                correct += (preds_labels == train_labels).sum().item()
                total += train_labels.shape[0]

            train_acc = round(correct / total, 4)

        print(f"I'am finished the epoch {epoch} evaluation on the training set")

        if (epoch + 1) % 50 == 0:
            print(f'epoch {epoch + 1}, training loss {running_loss}, train accuracy {train_acc}')

    # save the model at the end of 150 epochs
    model_path = f"{model_folder}/retrained_model_{epoch}.pt"

    torch.save(model, model_path)


def base_model1_10000_retraining():
    ### where to save the retrained model
    model_folder = '/home/wangc90/circRNA/circRNA_Data/model_outputs/Base_model1_retraining/Base_model1_retraining_10000'
    BS_LS_coordinates_path = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/updated_data/BS_LS_coordinates_final.csv'
    hg19_seq_dict_json_path = '/home/wangc90/circRNA/circRNA_Data/hg19_seq/hg19_seq_dict.json'
    flanking_dict_folder = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/flanking_dicts/'
    bs_ls_dataset = BS_LS_DataSet_Prep(BS_LS_coordinates_path=BS_LS_coordinates_path,
                                       hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                                       flanking_dict_folder=flanking_dict_folder,
                                       flanking_junction_bps=100,
                                       flanking_intron_bps=100,
                                       training_size=10000)

    ### generate the junction and flanking intron dict
    bs_ls_dataset.get_junction_flanking_intron_seq()

    ### use the 10000 for training RCM and junction seq and use 1000 for combine them
    train_key1, _, test_keys = bs_ls_dataset.get_train_test_keys()

    rcm_scores_folder = '/home/wangc90/Desktop/project_data/flanking_dicts/rcm_scores/'
    train_torch_upper_lower_features, train_torch_labels = bs_ls_dataset.seq_to_tensor(data_keys=train_key1,
                                                                                       rcm_folder=rcm_scores_folder,
                                                                                       is_rcm=False,
                                                                                       is_upper_lower_concat=True)

    BS_LS_dataset = BS_LS_upper_lower_concat_rcm(include_rcm=False,
                                                 seq_upper_lower_feature=train_torch_upper_lower_features,
                                                 flanking_rcm=None,
                                                 upper_rcm=None,
                                                 lower_rcm=None,
                                                 label=train_torch_labels)
    print(len(BS_LS_dataset))

    retraining(model=ConcatModel1_optuna_10000, dataset=BS_LS_dataset, model_folder=model_folder)

base_model1_10000_retraining()