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
from BS_LS_Training_Base_models import Objective, Objective_CV


### Model 1 input sequence 4 X 400 with 2CNN layer

class Model1_optuna(nn.Module):
    '''
        This model take in input sequence 4 X 400 with 2 CNN layer
    '''

    def __init__(self, trial):

        super(Model1_optuna, self).__init__()
        ### first CNN layer
        self.out_channel1 = trial.suggest_categorical('out_channel1', [32, 64, 128, 256, 512])
        kernel_size1 = trial.suggest_categorical('kernel_size1', [13, 15, 17, 19, 21])
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        self.maxpool1 = trial.suggest_categorical('maxpool1', [5, 10, 20])
        self.conv1_out_dim = 400 // self.maxpool1

        self.out_channel2 = trial.suggest_categorical('out_channel2', [32, 64, 128, 256, 512])
        kernel_size2 = trial.suggest_categorical('kernel_size2', [13, 15, 17, 19, 21])
        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                                   kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        self.maxpool2 = trial.suggest_categorical('maxpool2', [5, 10, 20])
        self.conv2_out_dim = 400 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class ConcatModel1_optuna(nn.Module):
    def __init__(self, trial):

        super(ConcatModel1_optuna, self).__init__()

        ### cnn for the concatenated sequence
        self.cnn = Model1_optuna(trial)

        # this is for two convlayer
        self.out_dim = self.cnn.conv2_out_dim
        self.out_channel = self.cnn.out_channel2

        self.fc1_input_dim = self.out_channel * self.out_dim

        self.fc1_out = trial.suggest_categorical('concat_fc1_out', [32, 64, 128, 256, 512])
        self.fc1 = nn.Linear(self.fc1_input_dim, self.fc1_out)

        self.fc1_bn = nn.BatchNorm1d(self.fc1_out)

        dropout_rate_fc1 = trial.suggest_categorical("concat_dropout_rate_fc1",  [0, 0.2, 0.4, 0.6, 0.8])
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc layer2
        # use dimension output with nn.CrossEntropyLoss()
        self.fc2_out = trial.suggest_categorical('concat_fc2_out', [8, 16, 32, 64, 128])
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

        self.fc2_bn = nn.BatchNorm1d(self.fc2_out)

        dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2",[0, 0.2, 0.4, 0.6, 0.8])

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


def base_model1_selection_optuna(num_trial):
    ### specify where to save the 3-fold CV validation acc
    val_acc_folder = '/home/wangc90/Desktop/project_result/Base_model1/val_acc_cv3'
    ### specify where to save the best model in the 3-fold CV
    model_folder = '/home/wangc90/Desktop/project_result/Base_model1/models'
    ### specify where to save the detailed optuna results
    optuna_folder = '/home/wangc90/Desktop/project_result/Base_model1/optuna'

    ## These need to be changed accordingly
    ## Specify where to find the BS_LS_coordinates_final.csv (can be downloaded from Data folder)
    BS_LS_coordinates_path = '/home/wangc90/Desktop/project_data/BS_LS_coordinates_final.csv'
    ## Specify where to find the json file containing hg19 sequence (can be downloaded from Data folder, see readme)
    hg19_seq_dict_json_path = '/home/wangc90/Desktop/project_data/hg19_seq_dict.json'
    ## (can be downloaded from Data folder, see readme)
    flanking_dict_folder = '/home/wangc90/Desktop/project_data/flanking_dicts/'
    bs_ls_dataset = BS_LS_DataSet_Prep(BS_LS_coordinates_path=BS_LS_coordinates_path,
                                       hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                                       flanking_dict_folder=flanking_dict_folder,
                                       flanking_junction_bps=100,
                                       flanking_intron_bps=100,
                                       training_size=10000)
    # This training_size can be changed, 10000, 9000, 8000 were used in this study

    ### generate the junction and flanking intron dict
    bs_ls_dataset.get_junction_flanking_intron_seq()
    ### use the 10000 for training RCM and junction seq and use remaining to combine the model
    train_key1, _, test_keys = bs_ls_dataset.get_train_test_keys()
    ### specify where to get the rcm_scores (can be downloaded from Data see readme)
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

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=10),
                                direction='maximize')

    study.optimize(Objective_CV(patience=5, cv=3, model=ConcatModel1_optuna,
                                dataset=BS_LS_dataset,
                                val_acc_folder=val_acc_folder,
                                model_folder=model_folder), n_trials=num_trial, gc_after_trial=True)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    with open(optuna_folder + '/optuna.txt', 'a') as f:
        f.write("Study statistics: \n")
        f.write(f"Number of finished trials: {len(study.trials)}\n")
        f.write(f"Number of pruned trials: {len(pruned_trials)}\n")
        f.write(f"Number of complete trials: {len(complete_trials)}\n")

        f.write("Best trial:\n")
        trial = study.best_trial
        f.write(f"Value: {trial.value}\n")
        f.write("Params:\n")
        for key, value in trial.params.items():
            f.write(f"{key}:{value}\n")

    df = study.trials_dataframe().drop(['state', 'datetime_start', 'datetime_complete', 'duration', 'number'], axis=1)
    df.to_csv(optuna_folder + '/optuna.csv', sep='\t', index=None)

### Conduct 500 Optuna trials for base model 1 with a given training size
base_model1_selection_optuna(500)