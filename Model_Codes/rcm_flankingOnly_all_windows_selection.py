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
from torchmetrics.classification import F1Score
import pickle

import sys
### import Dataset prepartion and model training classes from Auxiliary_Codes folder
from BS_LS_DataSet_3 import BS_LS_DataSet_Prep, RCM_Score
from BS_LS_Training_Base_models import Objective, Objective_CV


class RCM_optuna_flanking(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the flanking introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_flanking, self).__init__()

        # convlayer 1
        self.out_channel1 = trial.suggest_categorical('flanking_out_channel1', [128, 256, 512])
        #         self.out_channel1 = 128

        #         kernel_size1 = 5

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=self.out_channel1, \
                               kernel_size=5, stride=5, padding=0)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)

        self.out_channel2 = trial.suggest_categorical('flanking_out_channel2', [128, 256, 512])
        #         self.out_channel2 = 32

        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=5, stride=5, padding=0)

        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)

        self.conv2_out_dim = 10

        self.fc1_input_dim = self.conv2_out_dim * self.out_channel2

        #         print(f'fc1_input_dim: {self.fc1_input_dim}')

        self.fc1_out = trial.suggest_categorical('fc1_out', [128, 256, 512])
        #         self.fc1_out = 512

        # add the rcm feature dimension here as well (5*5+2)*3+2 = 83
        self.fc1 = nn.Linear(self.fc1_input_dim, self.fc1_out)

        self.fc1_bn = nn.BatchNorm1d(self.fc1_out)

        dropout_rate_fc1 = trial.suggest_categorical("concat_dropout_rate_fc1", [0, 0.1, 0.2, 0.4])
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc layer2
        # use dimension output with nn.CrossEntropyLoss()
        self.fc2_out = trial.suggest_categorical('fc2_out', [4, 8, 16, 32])
        #         self.fc2_out = 8
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

        self.fc2_bn = nn.BatchNorm1d(self.fc2_out)

        dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2", [0, 0.1, 0.2, 0.4])

        self.drop_nn2 = nn.Dropout(p=dropout_rate_fc2)

        self.fc3 = nn.Linear(self.fc2_out, 2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = torch.relu(self.conv2_bn(self.conv2(out)))

        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)

        out = self.fc1(out)
        out = self.drop_nn1(torch.relu(self.fc1_bn(out)))
        out = self.fc2(out)
        out = self.drop_nn2(torch.relu(self.fc2_bn(out)))
        out = self.fc3(out)
        return out


def rcm_flankingOnly_all_windows_optuna(num_trial):
    ## specify different kmer length to get the training data of the rcm score for that kmer
    ### just change this number to 10, 20, 40 and 80 to get the model performance for different kmer length

    study = optuna.create_study(direction='maximize')

    ### where to save the 3-fold CV validation acc based on the rcm score and mlp
    val_acc_folder = f'/home/wangc90/circRNA/circRNA_Data/model_outputs/rcm_flankingOnly_all_windows/8000/val_acc_cv3'
    ### where to save the best model in the 3-fold CV
    ### where to save the detailed optuna results
    optuna_folder = f'/home/wangc90/circRNA/circRNA_Data/model_outputs/rcm_flankingOnly_all_windows/8000/optuna'
    BS_LS_coordinates_path = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/updated_data/BS_LS_coordinates_final.csv'
    hg19_seq_dict_json_path = '/home/wangc90/circRNA/circRNA_Data/hg19_seq/hg19_seq_dict.json'
    flanking_dict_folder = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/flanking_dicts/'
    bs_ls_dataset = BS_LS_DataSet_Prep(BS_LS_coordinates_path=BS_LS_coordinates_path,
                                       hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                                       flanking_dict_folder=flanking_dict_folder,
                                       flanking_junction_bps=100,
                                       flanking_intron_bps=5000,
                                       training_size=8000)

    ## generate the junction and flanking intron dict
    bs_ls_dataset.get_junction_flanking_intron_seq()

    train_key_1, _, test_keys = bs_ls_dataset.get_train_test_keys()

    rcm_scores_folder = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/flanking_dicts/rcm_scores/'

    ### only use flanking_rcm_scores
    _, _, train_torch_flanking_rcm, _, \
    _, train_torch_labels = bs_ls_dataset.seq_to_tensor(data_keys=train_key_1, rcm_folder=rcm_scores_folder, \
                                                        is_rcm=True, is_upper_lower_concat=False)

    #     print(train_torch_flanking_rcm.shape)

    RCM_kmer_Score_dataset = RCM_Score(flanking_only=True,
                                       flanking_rcm=train_torch_flanking_rcm,
                                       upper_rcm=None, \
                                       lower_rcm=None, label=train_torch_labels)

    print(len(RCM_kmer_Score_dataset))

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=10),
                                direction='maximize')

    study.optimize(Objective_CV(cv=3, model=RCM_optuna_flanking,
                                dataset=RCM_kmer_Score_dataset,
                                val_acc_folder=val_acc_folder), n_trials=num_trial, gc_after_trial=True)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    with open(optuna_folder + '/optuna.txt', 'a') as f:
        f.write("Study statistiqcs: \n")
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

rcm_flankingOnly_all_windows_optuna(num_trial=500)