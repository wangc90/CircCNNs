import torch
import torch.optim as optim
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
from torchmetrics.classification import F1Score
import pickle
import sys
### import Dataset prepartion and model training classes from Auxiliary_Codes folder
from BS_LS_DataSet_3 import BS_LS_DataSet_Prep, RCM_Score
from BS_LS_Training_Base_models import Objective, Objective_CV


### retrain the RCM_triCNN model with selected hyperparameters
### 10000 training size used here for example

class RCM_optuna_flanking_10000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the flanking introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_flanking_10000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('flanking_out_channel1', [128, 256, 512])
        self.out_channel1 = 128

        #         kernel_size1 = 5

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=self.out_channel1, \
                               kernel_size=5, stride=5, padding=0)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)

        #         self.out_channel2 = trial.suggest_categorical('flanking_out_channel2', [128, 256, 512])
        self.out_channel2 = 128

        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=5, stride=5, padding=0)

        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)

        self.conv2_out_dim = 10

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = torch.relu(self.conv2_bn(self.conv2(out)))

        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class RCM_optuna_upper_10000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the upper introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_upper_10000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 512

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=self.out_channel1, \
                               kernel_size=5, stride=5, padding=0)

        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)

        #         self.out_channel2 = trial.suggest_categorical('upper_out_channel2', [128, 256, 512])
        self.out_channel2 = 256

        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=5, stride=5, padding=0)

        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        self.conv2_out_dim = 10

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = torch.relu(self.conv2_bn(self.conv2(out)))

        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class RCM_optuna_lower_10000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the lower introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_lower_10000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 512

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=self.out_channel1, \
                               kernel_size=5, stride=5, padding=0)

        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.out_channel2 = trial.suggest_categorical('lower_out_channel2', [128, 256, 512])
        self.out_channel2 = 512

        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=5, stride=5, padding=0)

        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        self.conv2_out_dim = 10

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = torch.relu(self.conv2_bn(self.conv2(out)))

        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class RCM_optuna_concate_10000(nn.Module):

    def __init__(self, trial):
        super(RCM_optuna_concate_10000, self).__init__()

        ### cnn for the flanking rcm scores
        self.rcm_flanking = RCM_optuna_flanking_10000(trial)

        self.flanking_out_dim = self.rcm_flanking.conv2_out_dim
        self.flanking_out_channel = self.rcm_flanking.out_channel2
        #         print(f'flanking out dim: {self.flanking_out_dim}, flanking out channel {self.flanking_out_channel}')

        ### cnn for the upper rcm scores
        self.rcm_upper = RCM_optuna_upper_10000(trial)

        self.upper_out_dim = self.rcm_upper.conv2_out_dim
        self.upper_out_channel = self.rcm_upper.out_channel2
        #         print(f'upper_out_dim: {self.upper_out_dim}, upper_out_channel {self.upper_out_channel}')

        ### cnn for the lower rcm scores
        self.rcm_lower = RCM_optuna_lower_10000(trial)

        self.lower_out_dim = self.rcm_lower.conv2_out_dim
        self.lower_out_channel = self.rcm_lower.out_channel2
        #         print(f'lower_out_dim: {self.lower_out_dim}, lower_out_channel {self.lower_out_channel}')

        self.fc1_input_dim = self.flanking_out_dim * self.flanking_out_channel + \
                             self.upper_out_dim * self.upper_out_channel + \
                             self.lower_out_dim * self.lower_out_channel

        #         print(f'fc1_input_dim: {self.fc1_input_dim}')

        #         self.fc1_out = trial.suggest_categorical('concat_fc1_out', [128, 256, 512])
        self.fc1_out = 512

        # add the rcm feature dimension here as well (5*5+2)*3+2 = 83
        self.rcm_concate_fc1 = nn.Linear(self.fc1_input_dim, self.fc1_out)

        self.rcm_concate_fc1_bn = nn.BatchNorm1d(self.fc1_out)

        #         dropout_rate_fc1 = trial.suggest_categorical("concat_dropout_rate_fc1",  [0, 0.1, 0.2, 0.4])
        dropout_rate_fc1 = 0.4
        self.drop_rcm_concate_fc1 = nn.Dropout(p=dropout_rate_fc1)

        # fc layer2
        # use dimension output with nn.CrossEntropyLoss()
        #         self.fc2_out = trial.suggest_categorical('concat_fc2_out', [4, 8, 16, 32])
        self.fc2_out = 8
        self.rcm_concate_fc2 = nn.Linear(self.fc1_out, self.fc2_out)

        self.rcm_concate_fc2_bn = nn.BatchNorm1d(self.fc2_out)

        #         dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2",[0, 0.1, 0.2, 0.4])
        dropout_rate_fc2 = 0.2

        self.drop_rcm_concate_fc2 = nn.Dropout(p=dropout_rate_fc2)

        self.fc3 = nn.Linear(self.fc2_out, 2)

    def forward(self, rcm_flanking, rcm_upper, rcm_lower):
        x1 = self.rcm_flanking(rcm_flanking)

        x2 = self.rcm_upper(rcm_upper)

        x3 = self.rcm_lower(rcm_lower)

        x = torch.cat((x1, x2, x3), dim=1)

        # feed the concatenated feature to fc1
        out = self.rcm_concate_fc1(x)

        out = self.drop_rcm_concate_fc1(torch.relu(self.rcm_concate_fc1_bn(out)))
        out = self.rcm_concate_fc2(out)
        out = self.drop_rcm_concate_fc2(torch.relu(self.rcm_concate_fc2_bn(out)))
        out = self.fc3(out)
        return out


def retraining(model, dataset, model_folder):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_size = 256

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #     print(len(train_loader))

    criterion = nn.CrossEntropyLoss()

    model = model('trial').to(device=device)
    #     print(model)

    optimizer_name = 'Adam'
    lr = 0.000016
    l2_lambda = 3.358313e-08
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


def RCM_triCNN_all_10000_retraining():
    ### where to save the 3-fold CV validation acc

    ### where to save the retrained model
    model_folder = '/home/wangc90/circRNA/circRNA_Data/model_outputs/RCM_triCNN_retraining/RCM_triCNN_retraining_10000'

    ## These need to be changed for redhawks
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

    ### use the 9000 for training RCM and junction seq and use 2000 for combine them
    train_key1, _, test_keys = bs_ls_dataset.get_train_test_keys()

    rcm_scores_folder = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/flanking_dicts/rcm_scores/'

    _, _, train_torch_flanking_rcm, train_torch_upper_rcm, train_torch_lower_rcm, \
    train_torch_labels = bs_ls_dataset.seq_to_tensor(data_keys=train_key1, \
                                                     rcm_folder=rcm_scores_folder, \
                                                     is_rcm=True, is_upper_lower_concat=False)

    RCM_kmer_Score_dataset = RCM_Score(flanking_only=False,
                                       flanking_rcm=train_torch_flanking_rcm, \
                                       upper_rcm=train_torch_upper_rcm, \
                                       lower_rcm=train_torch_lower_rcm, \
                                       label=train_torch_labels)
    print(len(RCM_kmer_Score_dataset))

    retraining(model=RCM_optuna_concate_10000, dataset=RCM_kmer_Score_dataset, model_folder=model_folder)

RCM_triCNN_all_10000_retraining()
