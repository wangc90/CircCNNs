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


class RCM_optuna_flanking_8000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the flanking introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_flanking_8000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('flanking_out_channel1', [128, 256, 512])
        self.out_channel1 = 512

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


class RCM_optuna_upper_8000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the upper introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_upper_8000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 128

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=self.out_channel1, \
                               kernel_size=5, stride=5, padding=0)

        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)

        #         self.out_channel2 = trial.suggest_categorical('upper_out_channel2', [128, 256, 512])
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


class RCM_optuna_lower_8000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the lower introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_lower_8000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 128

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


class RCM_optuna_concate_8000(nn.Module):
    ''''

    '''

    def __init__(self, trial):
        super(RCM_optuna_concate_8000, self).__init__()

        ### cnn for the flanking rcm scores
        self.rcm_flanking = RCM_optuna_flanking_8000(trial)

        self.flanking_out_dim = self.rcm_flanking.conv2_out_dim
        self.flanking_out_channel = self.rcm_flanking.out_channel2
        #         print(f'flanking out dim: {self.flanking_out_dim}, flanking out channel {self.flanking_out_channel}')

        ### cnn for the upper rcm scores
        self.rcm_upper = RCM_optuna_upper_8000(trial)

        self.upper_out_dim = self.rcm_upper.conv2_out_dim
        self.upper_out_channel = self.rcm_upper.out_channel2
        #         print(f'upper_out_dim: {self.upper_out_dim}, upper_out_channel {self.upper_out_channel}')

        ### cnn for the lower rcm scores
        self.rcm_lower = RCM_optuna_lower_8000(trial)

        self.lower_out_dim = self.rcm_lower.conv2_out_dim
        self.lower_out_channel = self.rcm_lower.out_channel2
        #         print(f'lower_out_dim: {self.lower_out_dim}, lower_out_channel {self.lower_out_channel}')

        self.fc1_input_dim = self.flanking_out_dim * self.flanking_out_channel + \
                             self.upper_out_dim * self.upper_out_channel + \
                             self.lower_out_dim * self.lower_out_channel

        #         print(f'fc1_input_dim: {self.fc1_input_dim}')

        #         self.fc1_out = trial.suggest_categorical('concat_fc1_out', [128, 256, 512])
        self.fc1_out = 256

        # add the rcm feature dimension here as well (5*5+2)*3+2 = 83
        self.rcm_concate_fc1 = nn.Linear(self.fc1_input_dim, self.fc1_out)

        self.rcm_concate_fc1_bn = nn.BatchNorm1d(self.fc1_out)

        #         dropout_rate_fc1 = trial.suggest_categorical("concat_dropout_rate_fc1",  [0, 0.1, 0.2, 0.4])
        concat_dropout_rate_fc1 = 0.1
        self.drop_rcm_concate_fc1 = nn.Dropout(p=concat_dropout_rate_fc1)

        # fc layer2
        # use dimension output with nn.CrossEntropyLoss()
        #         self.fc2_out = trial.suggest_categorical('concat_fc2_out', [4, 8, 16, 32])
        self.fc2_out = 16
        self.rcm_concate_fc2 = nn.Linear(self.fc1_out, self.fc2_out)

        self.rcm_concate_fc2_bn = nn.BatchNorm1d(self.fc2_out)

        #         dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2",[0, 0.1, 0.2, 0.4])
        concat_dropout_rate_fc2 = 0

        self.drop_rcm_concate_fc2 = nn.Dropout(p=concat_dropout_rate_fc2)

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


class RCM_optuna_flanking_9000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the flanking introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_flanking_9000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('flanking_out_channel1', [128, 256, 512])
        self.out_channel1 = 512

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


class RCM_optuna_upper_9000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the upper introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_upper_9000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 256

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


class RCM_optuna_lower_9000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the lower introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_lower_9000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 128

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


class RCM_optuna_concate_9000(nn.Module):
    ''''

    '''

    def __init__(self, trial):
        super(RCM_optuna_concate_9000, self).__init__()

        ### cnn for the flanking rcm scores
        self.rcm_flanking = RCM_optuna_flanking_9000(trial)

        self.flanking_out_dim = self.rcm_flanking.conv2_out_dim
        self.flanking_out_channel = self.rcm_flanking.out_channel2
        #         print(f'flanking out dim: {self.flanking_out_dim}, flanking out channel {self.flanking_out_channel}')

        ### cnn for the upper rcm scores
        self.rcm_upper = RCM_optuna_upper_9000(trial)

        self.upper_out_dim = self.rcm_upper.conv2_out_dim
        self.upper_out_channel = self.rcm_upper.out_channel2
        #         print(f'upper_out_dim: {self.upper_out_dim}, upper_out_channel {self.upper_out_channel}')

        ### cnn for the lower rcm scores
        self.rcm_lower = RCM_optuna_lower_9000(trial)

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
        dropout_rate_fc1 = 0.1
        self.drop_rcm_concate_fc1 = nn.Dropout(p=dropout_rate_fc1)

        # fc layer2
        # use dimension output with nn.CrossEntropyLoss()
        #         self.fc2_out = trial.suggest_categorical('concat_fc2_out', [4, 8, 16, 32])
        self.fc2_out = 32
        self.rcm_concate_fc2 = nn.Linear(self.fc1_out, self.fc2_out)

        self.rcm_concate_fc2_bn = nn.BatchNorm1d(self.fc2_out)

        #         dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2",[0, 0.1, 0.2, 0.4])
        dropout_rate_fc2 = 0

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
    ''''

    '''

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


class Model1_optuna_8000(nn.Module):
    '''
        This model take in input sequence 4 X 400 with 1 CNN layer
    '''

    def __init__(self, trial):
        super(Model1_optuna_8000, self).__init__()
        ### first CNN layer
        #         self.out_channel1 = trial.suggest_categorical('out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 13
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 400 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 13
        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('maxpool2', [5, 10, 20])
        self.maxpool2 = 5
        self.conv2_out_dim = 400 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class ConcatModel1_optuna_8000(nn.Module):
    def __init__(self, trial):
        super(ConcatModel1_optuna_8000, self).__init__()

        ### cnn for the concatenated sequence
        self.cnn = Model1_optuna_8000(trial)

        # this is for two convlayer
        self.out_dim = self.cnn.conv2_out_dim
        self.out_channel = self.cnn.out_channel2

        self.fc1_input_dim = self.out_channel * self.out_dim

        #         self.fc1_out = trial.suggest_categorical('concat_fc1_out', [128, 256, 512])
        self.fc1_out = 512
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
        dropout_rate_fc2 = 0.2

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


### Model 2 input sequence 4 X 200 + 4 X 200 with 1 or 2CNN layer
class Model2_optuna_upper_8000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_upper_8000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('upper_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 15

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('upper_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('upper_out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('upper_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21

        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('upper_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class Model2_optuna_lower_8000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_lower_8000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 256
        #         kernel_size1 = trial.suggest_categorical('lower_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 15
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('lower_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('lower_out_channel2', [128, 256, 512])
        self.out_channel2 = 128
        #         kernel_size2 = trial.suggest_categorical('lower_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21
        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('lower_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class ConcatModel2_optuna_8000(nn.Module):
    def __init__(self, trial):
        super(ConcatModel2_optuna_8000, self).__init__()
        ### cnn for the upper half sequence
        self.cnn_upper = Model2_optuna_upper_8000(trial)

        # this is for two convlayer
        self.upper_out_dim = self.cnn_upper.conv2_out_dim
        self.upper_out_channel = self.cnn_upper.out_channel2

        ### cnn for the lower half sequence
        self.cnn_lower = Model2_optuna_lower_8000(trial)

        # this is for two convlayer
        self.lower_out_dim = self.cnn_lower.conv2_out_dim
        self.lower_out_channel = self.cnn_lower.out_channel2

        self.upper_lower_concate_fc1_in = self.upper_out_channel * self.upper_out_dim + \
                                          self.lower_out_channel * self.lower_out_dim

        #         self.upper_lower_concate_fc1_out = trial.suggest_categorical('concat_fc1_out', [128, 256, 512])
        self.upper_lower_concate_fc1_out = 128

        self.upper_lower_concate_fc1 = nn.Linear(self.upper_lower_concate_fc1_in, self.upper_lower_concate_fc1_out)

        self.upper_lower_concate_fc1_bn = nn.BatchNorm1d(self.upper_lower_concate_fc1_out)

        #         dropout_rate_fc1 = trial.suggest_categorical("concat_dropout_rate_fc1",  [0, 0.1, 0.2, 0.4])
        dropout_rate_fc1 = 0.2
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc layer2
        # use dimension output with nn.CrossEntropyLoss()
        #         self.upper_lower_concate_fc2_out = trial.suggest_categorical('concat_fc2_out', [4, 8, 16, 32])
        self.upper_lower_concate_fc2_out = 8
        self.upper_lower_concate_fc2 = nn.Linear(self.upper_lower_concate_fc1_out, self.upper_lower_concate_fc2_out)
        self.upper_lower_concate_fc2_bn = nn.BatchNorm1d(self.upper_lower_concate_fc2_out)

        #         dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc2 = 0.1
        self.drop_nn2 = nn.Dropout(p=dropout_rate_fc2)

        self.upper_lower_concate_final = nn.Linear(self.upper_lower_concate_fc2_out, 2)

    def forward(self, seq_upper_feature, seq_lower_feature):
        # obatin the result from the cnn upper
        x1 = self.cnn_upper(seq_upper_feature)

        # obtain the result from the cnn lower
        x2 = self.cnn_lower(seq_lower_feature)

        x = torch.cat((x1, x2), dim=1)

        # feed the concatenated feature to fc1
        out = self.upper_lower_concate_fc1(x)
        out = self.drop_nn1(torch.relu(self.upper_lower_concate_fc1_bn(out)))

        out = self.upper_lower_concate_fc2(out)
        out = self.drop_nn2(torch.relu(self.upper_lower_concate_fc2_bn(out)))

        out = self.upper_lower_concate_final(out)

        return out


class Model1_optuna_9000(nn.Module):
    '''
        This model take in input sequence 4 X 400 with 1 CNN layer
    '''

    def __init__(self, trial):
        super(Model1_optuna_9000, self).__init__()
        ### first CNN layer
        #         self.out_channel1 = trial.suggest_categorical('out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 13
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
        self.maxpool2 = 5
        self.conv2_out_dim = 400 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class ConcatModel1_optuna_9000(nn.Module):
    def __init__(self, trial):
        super(ConcatModel1_optuna_9000, self).__init__()

        ### cnn for the concatenated sequence
        self.cnn = Model1_optuna_9000(trial)

        # this is for two convlayer
        self.out_dim = self.cnn.conv2_out_dim
        self.out_channel = self.cnn.out_channel2

        self.fc1_input_dim = self.out_channel * self.out_dim

        #         self.fc1_out = trial.suggest_categorical('concat_fc1_out', [128, 256, 512])
        self.fc1_out = 512
        self.fc1 = nn.Linear(self.fc1_input_dim, self.fc1_out)

        self.fc1_bn = nn.BatchNorm1d(self.fc1_out)

        #         dropout_rate_fc1 = trial.suggest_categorical("concat_dropout_rate_fc1",  [0, 0.1, 0.2, 0.4])
        dropout_rate_fc1 = 0.2
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc layer2
        # use dimension output with nn.CrossEntropyLoss()
        #         self.fc2_out = trial.suggest_categorical('concat_fc2_out', [4, 8, 16, 32])
        self.fc2_out = 16
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


### Model 2 input sequence 4 X 200 + 4 X 200 with 1 or 2CNN layer
class Model2_optuna_upper_9000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_upper_9000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('upper_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 21

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('upper_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('upper_out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('upper_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21

        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('upper_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class Model2_optuna_lower_9000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_lower_9000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('lower_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 13
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('lower_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('lower_out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('lower_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21
        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('lower_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class ConcatModel2_optuna_9000(nn.Module):
    def __init__(self, trial):
        super(ConcatModel2_optuna_9000, self).__init__()
        ### cnn for the upper half sequence
        self.cnn_upper = Model2_optuna_upper_9000(trial)

        # this is for two convlayer
        self.upper_out_dim = self.cnn_upper.conv2_out_dim
        self.upper_out_channel = self.cnn_upper.out_channel2

        ### cnn for the lower half sequence
        self.cnn_lower = Model2_optuna_lower_9000(trial)

        # this is for two convlayer
        self.lower_out_dim = self.cnn_lower.conv2_out_dim
        self.lower_out_channel = self.cnn_lower.out_channel2

        self.upper_lower_concate_fc1_in = self.upper_out_channel * self.upper_out_dim + \
                                          self.lower_out_channel * self.lower_out_dim

        #         self.upper_lower_concate_fc1_out = trial.suggest_categorical('concat_fc1_out', [128, 256, 512])
        self.upper_lower_concate_fc1_out = 128

        self.upper_lower_concate_fc1 = nn.Linear(self.upper_lower_concate_fc1_in, self.upper_lower_concate_fc1_out)

        self.upper_lower_concate_fc1_bn = nn.BatchNorm1d(self.upper_lower_concate_fc1_out)

        #         dropout_rate_fc1 = trial.suggest_categorical("concat_dropout_rate_fc1",  [0, 0.1, 0.2, 0.4])
        dropout_rate_fc1 = 0.2
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc layer2
        # use dimension output with nn.CrossEntropyLoss()
        #         self.upper_lower_concate_fc2_out = trial.suggest_categorical('concat_fc2_out', [4, 8, 16, 32])
        self.upper_lower_concate_fc2_out = 4
        self.upper_lower_concate_fc2 = nn.Linear(self.upper_lower_concate_fc1_out, self.upper_lower_concate_fc2_out)
        self.upper_lower_concate_fc2_bn = nn.BatchNorm1d(self.upper_lower_concate_fc2_out)

        #         dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc2 = 0.1
        self.drop_nn2 = nn.Dropout(p=dropout_rate_fc2)

        self.upper_lower_concate_final = nn.Linear(self.upper_lower_concate_fc2_out, 2)

    def forward(self, seq_upper_feature, seq_lower_feature):
        # obatin the result from the cnn upper
        x1 = self.cnn_upper(seq_upper_feature)

        # obtain the result from the cnn lower
        x2 = self.cnn_lower(seq_lower_feature)

        x = torch.cat((x1, x2), dim=1)

        # feed the concatenated feature to fc1
        out = self.upper_lower_concate_fc1(x)
        out = self.drop_nn1(torch.relu(self.upper_lower_concate_fc1_bn(out)))

        out = self.upper_lower_concate_fc2(out)
        out = self.drop_nn2(torch.relu(self.upper_lower_concate_fc2_bn(out)))

        out = self.upper_lower_concate_final(out)

        return out


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


### Model 2 input sequence 4 X 200 + 4 X 200 with 1 or 2CNN layer
class Model2_optuna_upper_10000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_upper_10000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('upper_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 15

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('upper_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('upper_out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('upper_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21

        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('upper_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class Model2_optuna_lower_10000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_lower_10000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 256
        #         kernel_size1 = trial.suggest_categorical('lower_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 13
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('lower_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('lower_out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('lower_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21
        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('lower_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class ConcatModel2_optuna_10000(nn.Module):
    def __init__(self, trial):
        super(ConcatModel2_optuna_10000, self).__init__()
        ### cnn for the upper half sequence
        self.cnn_upper = Model2_optuna_upper_10000(trial)

        # this is for two convlayer
        self.upper_out_dim = self.cnn_upper.conv2_out_dim
        self.upper_out_channel = self.cnn_upper.out_channel2

        ### cnn for the lower half sequence
        self.cnn_lower = Model2_optuna_lower_10000(trial)

        # this is for two convlayer
        self.lower_out_dim = self.cnn_lower.conv2_out_dim
        self.lower_out_channel = self.cnn_lower.out_channel2

        self.upper_lower_concate_fc1_in = self.upper_out_channel * self.upper_out_dim + \
                                          self.lower_out_channel * self.lower_out_dim

        #         self.upper_lower_concate_fc1_out = trial.suggest_categorical('concat_fc1_out', [128, 256, 512])
        self.upper_lower_concate_fc1_out = 512

        self.upper_lower_concate_fc1 = nn.Linear(self.upper_lower_concate_fc1_in, self.upper_lower_concate_fc1_out)

        self.upper_lower_concate_fc1_bn = nn.BatchNorm1d(self.upper_lower_concate_fc1_out)

        #         dropout_rate_fc1 = trial.suggest_categorical("concat_dropout_rate_fc1",  [0, 0.1, 0.2, 0.4])
        dropout_rate_fc1 = 0
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc layer2
        # use dimension output with nn.CrossEntropyLoss()
        #         self.upper_lower_concate_fc2_out = trial.suggest_categorical('concat_fc2_out', [4, 8, 16, 32])
        self.upper_lower_concate_fc2_out = 4
        self.upper_lower_concate_fc2 = nn.Linear(self.upper_lower_concate_fc1_out, self.upper_lower_concate_fc2_out)
        self.upper_lower_concate_fc2_bn = nn.BatchNorm1d(self.upper_lower_concate_fc2_out)

        #         dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc2 = 0
        self.drop_nn2 = nn.Dropout(p=dropout_rate_fc2)

        self.upper_lower_concate_final = nn.Linear(self.upper_lower_concate_fc2_out, 2)

    def forward(self, seq_upper_feature, seq_lower_feature):
        # obatin the result from the cnn upper
        x1 = self.cnn_upper(seq_upper_feature)

        # obtain the result from the cnn lower
        x2 = self.cnn_lower(seq_lower_feature)

        x = torch.cat((x1, x2), dim=1)

        # feed the concatenated feature to fc1
        out = self.upper_lower_concate_fc1(x)
        out = self.drop_nn1(torch.relu(self.upper_lower_concate_fc1_bn(out)))

        out = self.upper_lower_concate_fc2(out)
        out = self.drop_nn2(torch.relu(self.upper_lower_concate_fc2_bn(out)))

        out = self.upper_lower_concate_final(out)

        return out


#### get the model structure to flatten layers first

class Model2_optuna_upper_8000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_upper_8000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('upper_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 15

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('upper_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('upper_out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('upper_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21

        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('upper_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class Model2_optuna_lower_8000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_lower_8000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 256
        #         kernel_size1 = trial.suggest_categorical('lower_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 15
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('lower_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('lower_out_channel2', [128, 256, 512])
        self.out_channel2 = 128
        #         kernel_size2 = trial.suggest_categorical('lower_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21
        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('lower_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class RCM_optuna_flanking_8000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the flanking introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_flanking_8000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('flanking_out_channel1', [128, 256, 512])
        self.out_channel1 = 512

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


class RCM_optuna_upper_8000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the upper introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_upper_8000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 128

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=self.out_channel1, \
                               kernel_size=5, stride=5, padding=0)

        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)

        #         self.out_channel2 = trial.suggest_categorical('upper_out_channel2', [128, 256, 512])
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


class RCM_optuna_lower_8000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the lower introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_lower_8000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 128

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


class Base2_RCM_triCNN_combined_8000(nn.Module):

    def __init__(self, trial):
        super(Base2_RCM_triCNN_combined_8000, self).__init__()

        ### cnns for model2
        self.cnn_upper = Model2_optuna_upper_8000(trial)

        # this is for two convlayer
        self.upper_out_dim = self.cnn_upper.conv2_out_dim
        self.upper_out_channel = self.cnn_upper.out_channel2

        ### cnn for the lower half sequence
        self.cnn_lower = Model2_optuna_lower_8000(trial)

        # this is for two convlayer
        self.lower_out_dim = self.cnn_lower.conv2_out_dim
        self.lower_out_channel = self.cnn_lower.out_channel2

        ### cnn for RCM features
        self.rcm_flanking = RCM_optuna_flanking_8000(trial)
        self.rcm_upper = RCM_optuna_upper_8000(trial)
        self.rcm_lower = RCM_optuna_lower_8000(trial)

        ### fc1 layers from the ConcatModel2_optuna_8000

        self.upper_lower_concate_fc1_in = self.upper_out_channel * self.upper_out_dim + \
                                          self.lower_out_channel * self.lower_out_dim

        self.upper_lower_concate_fc1_out = 128

        self.upper_lower_concate_fc1 = nn.Linear(self.upper_lower_concate_fc1_in, self.upper_lower_concate_fc1_out)

        self.upper_lower_concate_fc1_bn = nn.BatchNorm1d(self.upper_lower_concate_fc1_out)

        dropout_rate_fc1 = 0.2
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc2 layers from the ConcatModel2_optuna_8000

        self.upper_lower_concate_fc2_out = 8
        self.upper_lower_concate_fc2 = nn.Linear(self.upper_lower_concate_fc1_out, self.upper_lower_concate_fc2_out)
        self.upper_lower_concate_fc2_bn = nn.BatchNorm1d(self.upper_lower_concate_fc2_out)

        #         dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc2 = 0.1
        self.drop_nn2 = nn.Dropout(p=dropout_rate_fc2)

        ### fc1 layers from the RCM_optuna_concate_8000
        self.rcm_concate_fc1_out = 256
        self.rcm_concate_fc1_in = self.rcm_flanking.conv2_out_dim * self.rcm_flanking.out_channel2 + \
                                  self.rcm_upper.conv2_out_dim * self.rcm_upper.out_channel2 + \
                                  self.rcm_lower.conv2_out_dim * self.rcm_lower.out_channel2

        self.rcm_concate_fc1 = nn.Linear(self.rcm_concate_fc1_in, self.rcm_concate_fc1_out)

        self.rcm_concate_fc1_bn = nn.BatchNorm1d(self.rcm_concate_fc1_out)

        dropout_rate_rcm_concate_fc1 = 0.1
        self.drop_rcm_concate_fc1 = nn.Dropout(p=dropout_rate_rcm_concate_fc1)

        ### fc2 layers from the RCM_optuna_concate_8000
        self.rcm_concate_fc2_out = 16
        self.rcm_concate_fc2 = nn.Linear(self.rcm_concate_fc1_out, self.rcm_concate_fc2_out)

        self.rcm_concate_fc2_bn = nn.BatchNorm1d(self.rcm_concate_fc2_out)

        dropout_rate_rcm_concate_fc2 = 0
        self.drop_rcm_concate_fc2 = nn.Dropout(p=dropout_rate_rcm_concate_fc2)

        ## newly added two layers

        #         self.fc3_out = trial.suggest_categorical('Base2_RCM_triCNN_combined_8000_fc3_out', [8, 16, 32])
        self.fc3_out = 32
        self.fc3 = nn.Linear(self.rcm_concate_fc2_out + self.upper_lower_concate_fc2_out, self.fc3_out)
        self.fc3_bn = nn.BatchNorm1d(self.fc3_out)
        #         dropout_rate_fc3 = trial.suggest_categorical("Base2_RCM_triCNN_combined_8000_dropout_rate_fc3", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc3 = 0
        self.drop_fc3 = nn.Dropout(p=dropout_rate_fc3)

        #         self.fc4_out = trial.suggest_categorical('Base2_RCM_triCNN_combined_8000_fc4_out', [4, 8, 16])
        self.fc4_out = 8
        self.fc4 = nn.Linear(self.fc3_out, self.fc4_out)
        self.fc4_bn = nn.BatchNorm1d(self.fc4_out)
        #         dropout_rate_fc4 = 0
        #         dropout_rate_fc4 = trial.suggest_categorical("Base2_RCM_triCNN_combined_8000_dropout_rate_fc4", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc4 = 0.2
        self.drop_fc4 = nn.Dropout(p=dropout_rate_fc4)

        self.fc5_out = 2
        self.fc5 = nn.Linear(self.fc4_out, self.fc5_out)

    def forward(self, seq_upper_feature, seq_lower_feature, rcm_flanking, rcm_upper, rcm_lower):
        ### layer to process junction seq
        x1 = self.cnn_upper(seq_upper_feature)
        x2 = self.cnn_lower(seq_lower_feature)

        x_12 = torch.cat((x1, x2), dim=1)

        x_12 = self.upper_lower_concate_fc1(x_12)
        x_12 = self.drop_nn1(torch.relu(self.upper_lower_concate_fc1_bn(x_12)))

        x_12 = self.upper_lower_concate_fc2(x_12)
        x_12 = self.drop_nn2(torch.relu(self.upper_lower_concate_fc2_bn(x_12)))

        ### layer to process RCM information
        x3 = self.rcm_flanking(rcm_flanking)
        x4 = self.rcm_upper(rcm_upper)
        x5 = self.rcm_lower(rcm_lower)

        x_345 = torch.cat((x3, x4, x5), dim=1)
        x_345 = self.rcm_concate_fc1(x_345)

        x_345 = self.drop_rcm_concate_fc1(torch.relu(self.rcm_concate_fc1_bn(x_345)))

        x_345 = self.rcm_concate_fc2(x_345)
        x_345 = self.drop_rcm_concate_fc2(torch.relu(self.rcm_concate_fc2_bn(x_345)))

        x_12345 = F.normalize(torch.cat((x_12, x_345), dim=1), dim=1)

        out = self.drop_fc3(torch.relu(self.fc3_bn(self.fc3(x_12345))))
        out = self.drop_fc4(torch.relu(self.fc4_bn(self.fc4(out))))

        out = self.fc5(out)

        return out


#### get the model structure to flatten layers first

class Model2_optuna_upper_9000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_upper_9000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('upper_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 21

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('upper_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('upper_out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('upper_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21

        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('upper_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class Model2_optuna_lower_9000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_lower_9000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('lower_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 13
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('lower_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('lower_out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('lower_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21
        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('lower_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class RCM_optuna_flanking_9000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the flanking introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_flanking_9000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('flanking_out_channel1', [128, 256, 512])
        self.out_channel1 = 512

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


class RCM_optuna_upper_9000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the upper introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_upper_9000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 256

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


class RCM_optuna_lower_9000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the lower introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_lower_9000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 128

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


class Base2_RCM_triCNN_combined_9000(nn.Module):

    def __init__(self, trial):
        super(Base2_RCM_triCNN_combined_9000, self).__init__()

        ### cnns for model2
        self.cnn_upper = Model2_optuna_upper_9000(trial)

        # this is for two convlayer
        self.upper_out_dim = self.cnn_upper.conv2_out_dim
        self.upper_out_channel = self.cnn_upper.out_channel2

        ### cnn for the lower half sequence
        self.cnn_lower = Model2_optuna_lower_9000(trial)

        # this is for two convlayer
        self.lower_out_dim = self.cnn_lower.conv2_out_dim
        self.lower_out_channel = self.cnn_lower.out_channel2

        ### cnn for RCM features
        self.rcm_flanking = RCM_optuna_flanking_9000(trial)
        self.rcm_upper = RCM_optuna_upper_9000(trial)
        self.rcm_lower = RCM_optuna_lower_9000(trial)

        ### fc1 layers from the ConcatModel2_optuna_9000

        self.upper_lower_concate_fc1_in = self.upper_out_channel * self.upper_out_dim + \
                                          self.lower_out_channel * self.lower_out_dim

        self.upper_lower_concate_fc1_out = 128

        self.upper_lower_concate_fc1 = nn.Linear(self.upper_lower_concate_fc1_in, self.upper_lower_concate_fc1_out)

        self.upper_lower_concate_fc1_bn = nn.BatchNorm1d(self.upper_lower_concate_fc1_out)

        dropout_rate_fc1 = 0.2
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc2 layers from the ConcatModel2_optuna_9000

        self.upper_lower_concate_fc2_out = 4
        self.upper_lower_concate_fc2 = nn.Linear(self.upper_lower_concate_fc1_out, self.upper_lower_concate_fc2_out)
        self.upper_lower_concate_fc2_bn = nn.BatchNorm1d(self.upper_lower_concate_fc2_out)

        #         dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc2 = 0.1
        self.drop_nn2 = nn.Dropout(p=dropout_rate_fc2)

        ### fc1 layers from the RCM_optuna_concate_8000
        self.rcm_concate_fc1_out = 512
        self.rcm_concate_fc1_in = self.rcm_flanking.conv2_out_dim * self.rcm_flanking.out_channel2 + \
                                  self.rcm_upper.conv2_out_dim * self.rcm_upper.out_channel2 + \
                                  self.rcm_lower.conv2_out_dim * self.rcm_lower.out_channel2

        self.rcm_concate_fc1 = nn.Linear(self.rcm_concate_fc1_in, self.rcm_concate_fc1_out)

        self.rcm_concate_fc1_bn = nn.BatchNorm1d(self.rcm_concate_fc1_out)

        dropout_rate_rcm_concate_fc1 = 0.1
        self.drop_rcm_concate_fc1 = nn.Dropout(p=dropout_rate_rcm_concate_fc1)

        ### fc2 layers from the RCM_optuna_concate_8000
        self.rcm_concate_fc2_out = 32
        self.rcm_concate_fc2 = nn.Linear(self.rcm_concate_fc1_out, self.rcm_concate_fc2_out)

        self.rcm_concate_fc2_bn = nn.BatchNorm1d(self.rcm_concate_fc2_out)

        dropout_rate_rcm_concate_fc2 = 0
        self.drop_rcm_concate_fc2 = nn.Dropout(p=dropout_rate_rcm_concate_fc2)

        ## newly added two layers

        #         self.fc3_out = trial.suggest_categorical('Base2_RCM_triCNN_combined_9000_fc3_out', [8, 16, 32])
        self.fc3_out = 16
        self.fc3 = nn.Linear(self.rcm_concate_fc2_out + self.upper_lower_concate_fc2_out, self.fc3_out)
        self.fc3_bn = nn.BatchNorm1d(self.fc3_out)
        #         dropout_rate_fc3 = trial.suggest_categorical("Base2_RCM_triCNN_combined_9000_dropout_rate_fc3", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc3 = 0.4
        self.drop_fc3 = nn.Dropout(p=dropout_rate_fc3)

        #         self.fc4_out = trial.suggest_categorical('Base2_RCM_triCNN_combined_9000_fc4_out', [4, 8, 16])
        self.fc4_out = 4
        self.fc4 = nn.Linear(self.fc3_out, self.fc4_out)
        self.fc4_bn = nn.BatchNorm1d(self.fc4_out)
        dropout_rate_fc4 = 0.2
        #         dropout_rate_fc4 = trial.suggest_categorical("Base2_RCM_triCNN_combined_9000_dropout_rate_fc4", [0, 0.1, 0.2, 0.4])
        self.drop_fc4 = nn.Dropout(p=dropout_rate_fc4)

        self.fc5_out = 2
        self.fc5 = nn.Linear(self.fc4_out, self.fc5_out)

    def forward(self, seq_upper_feature, seq_lower_feature, rcm_flanking, rcm_upper, rcm_lower):
        ### layer to process junction seq
        x1 = self.cnn_upper(seq_upper_feature)
        x2 = self.cnn_lower(seq_lower_feature)

        x_12 = torch.cat((x1, x2), dim=1)

        x_12 = self.upper_lower_concate_fc1(x_12)
        x_12 = self.drop_nn1(torch.relu(self.upper_lower_concate_fc1_bn(x_12)))

        x_12 = self.upper_lower_concate_fc2(x_12)
        x_12 = self.drop_nn2(torch.relu(self.upper_lower_concate_fc2_bn(x_12)))

        ### layer to process RCM information
        x3 = self.rcm_flanking(rcm_flanking)
        x4 = self.rcm_upper(rcm_upper)
        x5 = self.rcm_lower(rcm_lower)

        x_345 = torch.cat((x3, x4, x5), dim=1)
        x_345 = self.rcm_concate_fc1(x_345)

        x_345 = self.drop_rcm_concate_fc1(torch.relu(self.rcm_concate_fc1_bn(x_345)))

        x_345 = self.rcm_concate_fc2(x_345)
        x_345 = self.drop_rcm_concate_fc2(torch.relu(self.rcm_concate_fc2_bn(x_345)))

        x_12345 = F.normalize(torch.cat((x_12, x_345), dim=1), dim=1)

        out = self.drop_fc3(torch.relu(self.fc3_bn(self.fc3(x_12345))))
        out = self.drop_fc4(torch.relu(self.fc4_bn(self.fc4(out))))

        out = self.fc5(out)

        return out


#### get the model structure to flatten layers first

## Model 2 input sequence 4 X 200 + 4 X 200 with 1 or 2CNN layer
class Model2_optuna_upper_10000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_upper_10000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('upper_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 15

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('upper_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('upper_out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('upper_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21

        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('upper_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class Model2_optuna_lower_10000(nn.Module):
    '''
        This is for 2-d model to process the upper half of the sequence with 1 or 2 CNN
    '''

    def __init__(self, trial):
        super(Model2_optuna_lower_10000, self).__init__()
        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 256
        #         kernel_size1 = trial.suggest_categorical('lower_kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 13
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('lower_maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 200 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('lower_out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('lower_kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 21
        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('lower_maxpool2', [5, 10])
        self.maxpool2 = 10
        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


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


class Base2_RCM_triCNN_combined_10000(nn.Module):

    def __init__(self, trial):
        super(Base2_RCM_triCNN_combined_10000, self).__init__()

        ### cnns for model2
        self.cnn_upper = Model2_optuna_upper_10000(trial)

        # this is for two convlayer
        self.upper_out_dim = self.cnn_upper.conv2_out_dim
        self.upper_out_channel = self.cnn_upper.out_channel2

        ### cnn for the lower half sequence
        self.cnn_lower = Model2_optuna_lower_10000(trial)

        # this is for two convlayer
        self.lower_out_dim = self.cnn_lower.conv2_out_dim
        self.lower_out_channel = self.cnn_lower.out_channel2

        ### cnn for RCM features
        self.rcm_flanking = RCM_optuna_flanking_10000(trial)
        self.rcm_upper = RCM_optuna_upper_10000(trial)
        self.rcm_lower = RCM_optuna_lower_10000(trial)

        ### fc1 layers from the ConcatModel2_optuna_10000

        self.upper_lower_concate_fc1_in = self.upper_out_channel * self.upper_out_dim + \
                                          self.lower_out_channel * self.lower_out_dim

        self.upper_lower_concate_fc1_out = 512

        self.upper_lower_concate_fc1 = nn.Linear(self.upper_lower_concate_fc1_in, self.upper_lower_concate_fc1_out)

        self.upper_lower_concate_fc1_bn = nn.BatchNorm1d(self.upper_lower_concate_fc1_out)

        dropout_rate_fc1 = 0
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc2 layers from the ConcatModel2_optuna_9000

        self.upper_lower_concate_fc2_out = 4
        self.upper_lower_concate_fc2 = nn.Linear(self.upper_lower_concate_fc1_out, self.upper_lower_concate_fc2_out)
        self.upper_lower_concate_fc2_bn = nn.BatchNorm1d(self.upper_lower_concate_fc2_out)

        #         dropout_rate_fc2 = trial.suggest_categorical("concat_dropout_rate_fc2", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc2 = 0
        self.drop_nn2 = nn.Dropout(p=dropout_rate_fc2)

        ### fc1 layers from the RCM_optuna_concate_8000
        self.rcm_concate_fc1_out = 512
        self.rcm_concate_fc1_in = self.rcm_flanking.conv2_out_dim * self.rcm_flanking.out_channel2 + \
                                  self.rcm_upper.conv2_out_dim * self.rcm_upper.out_channel2 + \
                                  self.rcm_lower.conv2_out_dim * self.rcm_lower.out_channel2

        self.rcm_concate_fc1 = nn.Linear(self.rcm_concate_fc1_in, self.rcm_concate_fc1_out)

        self.rcm_concate_fc1_bn = nn.BatchNorm1d(self.rcm_concate_fc1_out)

        dropout_rate_rcm_concate_fc1 = 0.4
        self.drop_rcm_concate_fc1 = nn.Dropout(p=dropout_rate_rcm_concate_fc1)

        ### fc2 layers from the RCM_optuna_concate_8000
        self.rcm_concate_fc2_out = 8
        self.rcm_concate_fc2 = nn.Linear(self.rcm_concate_fc1_out, self.rcm_concate_fc2_out)

        self.rcm_concate_fc2_bn = nn.BatchNorm1d(self.rcm_concate_fc2_out)

        dropout_rate_rcm_concate_fc2 = 0.2
        self.drop_rcm_concate_fc2 = nn.Dropout(p=dropout_rate_rcm_concate_fc2)

        ## newly added two layers

        #         self.fc3_out = trial.suggest_categorical('Base2_RCM_triCNN_combined_10000_fc3_out', [8, 16, 32])
        self.fc3_out = 16
        self.fc3 = nn.Linear(self.rcm_concate_fc2_out + self.upper_lower_concate_fc2_out, self.fc3_out)
        self.fc3_bn = nn.BatchNorm1d(self.fc3_out)
        #         dropout_rate_fc3 = trial.suggest_categorical("Base2_RCM_triCNN_combined_10000_dropout_rate_fc3", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc3 = 0.4
        self.drop_fc3 = nn.Dropout(p=dropout_rate_fc3)

        #         self.fc4_out = trial.suggest_categorical('Base2_RCM_triCNN_combined_10000_fc4_out', [4, 8, 16])
        self.fc4_out = 16
        self.fc4 = nn.Linear(self.fc3_out, self.fc4_out)
        self.fc4_bn = nn.BatchNorm1d(self.fc4_out)
        dropout_rate_fc4 = 0.1
        #         dropout_rate_fc4 = trial.suggest_categorical("Base2_RCM_triCNN_combined_10000_dropout_rate_fc4", [0, 0.1, 0.2, 0.4])
        self.drop_fc4 = nn.Dropout(p=dropout_rate_fc4)

        self.fc5_out = 2
        self.fc5 = nn.Linear(self.fc4_out, self.fc5_out)

    def forward(self, seq_upper_feature, seq_lower_feature, rcm_flanking, rcm_upper, rcm_lower):
        ### layer to process junction seq
        x1 = self.cnn_upper(seq_upper_feature)
        x2 = self.cnn_lower(seq_lower_feature)

        x_12 = torch.cat((x1, x2), dim=1)

        x_12 = self.upper_lower_concate_fc1(x_12)
        x_12 = self.drop_nn1(torch.relu(self.upper_lower_concate_fc1_bn(x_12)))

        x_12 = self.upper_lower_concate_fc2(x_12)
        x_12 = self.drop_nn2(torch.relu(self.upper_lower_concate_fc2_bn(x_12)))

        ### layer to process RCM information
        x3 = self.rcm_flanking(rcm_flanking)
        x4 = self.rcm_upper(rcm_upper)
        x5 = self.rcm_lower(rcm_lower)

        x_345 = torch.cat((x3, x4, x5), dim=1)
        x_345 = self.rcm_concate_fc1(x_345)

        x_345 = self.drop_rcm_concate_fc1(torch.relu(self.rcm_concate_fc1_bn(x_345)))

        x_345 = self.rcm_concate_fc2(x_345)
        x_345 = self.drop_rcm_concate_fc2(torch.relu(self.rcm_concate_fc2_bn(x_345)))

        x_12345 = F.normalize(torch.cat((x_12, x_345), dim=1), dim=1)

        out = self.drop_fc3(torch.relu(self.fc3_bn(self.fc3(x_12345))))
        out = self.drop_fc4(torch.relu(self.fc4_bn(self.fc4(out))))

        out = self.fc5(out)

        return out


#### get the model structure to flatten layers first

class Model1_optuna_9000(nn.Module):
    '''
        This model take in input sequence 4 X 400 with 1 CNN layer
    '''

    def __init__(self, trial):
        super(Model1_optuna_9000, self).__init__()
        ### first CNN layer
        #         self.out_channel1 = trial.suggest_categorical('out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 13
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
        self.maxpool2 = 5
        self.conv2_out_dim = 400 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class RCM_optuna_flanking_9000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the flanking introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_flanking_9000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('flanking_out_channel1', [128, 256, 512])
        self.out_channel1 = 512

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


class RCM_optuna_upper_9000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the upper introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_upper_9000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 256

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


class RCM_optuna_lower_9000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the lower introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_lower_9000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 128

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


class Base1_RCM_triCNN_combined_9000(nn.Module):

    def __init__(self, trial):
        super(Base1_RCM_triCNN_combined_9000, self).__init__()

        ### cnn for model1

        self.cnn = Model1_optuna_9000(trial)

        ### cnn for RCM features
        self.rcm_flanking = RCM_optuna_flanking_9000(trial)
        self.rcm_upper = RCM_optuna_upper_9000(trial)
        self.rcm_lower = RCM_optuna_lower_9000(trial)

        ### fc1 layers from the ConcatModel1_optuna_9000

        self.fc1_input_dim = self.cnn.conv2_out_dim * self.cnn.out_channel2

        self.fc1_out = 512

        self.fc1 = nn.Linear(self.fc1_input_dim, self.fc1_out)

        self.fc1_bn = nn.BatchNorm1d(self.fc1_out)

        dropout_rate_fc1 = 0.2
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc2 layers from the ConcatModel1_optuna_8000

        self.fc2_out = 16
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

        self.fc2_bn = nn.BatchNorm1d(self.fc2_out)

        dropout_rate_fc2 = 0
        self.drop_nn2 = nn.Dropout(p=dropout_rate_fc2)

        ### fc1 layers from the RCM_optuna_concate_8000
        self.rcm_concate_fc1_out = 512
        self.rcm_concate_fc1_in = self.rcm_flanking.conv2_out_dim * self.rcm_flanking.out_channel2 + \
                                  self.rcm_upper.conv2_out_dim * self.rcm_upper.out_channel2 + \
                                  self.rcm_lower.conv2_out_dim * self.rcm_lower.out_channel2

        self.rcm_concate_fc1 = nn.Linear(self.rcm_concate_fc1_in, self.rcm_concate_fc1_out)

        self.rcm_concate_fc1_bn = nn.BatchNorm1d(self.rcm_concate_fc1_out)

        dropout_rate_rcm_concate_fc1 = 0.1
        self.drop_rcm_concate_fc1 = nn.Dropout(p=dropout_rate_rcm_concate_fc1)

        ### fc2 layers from the RCM_optuna_concate_8000
        self.rcm_concate_fc2_out = 32
        self.rcm_concate_fc2 = nn.Linear(self.rcm_concate_fc1_out, self.rcm_concate_fc2_out)

        self.rcm_concate_fc2_bn = nn.BatchNorm1d(self.rcm_concate_fc2_out)

        dropout_rate_rcm_concate_fc2 = 0
        self.drop_rcm_concate_fc2 = nn.Dropout(p=dropout_rate_rcm_concate_fc2)

        ## newly added two layers

        #         self.fc3_out = trial.suggest_categorical('Base1_RCM_triCNN_combined_9000_fc3_out',[8, 16, 32])
        self.fc3_out = 16
        self.fc3 = nn.Linear(self.rcm_concate_fc2_out + self.fc2_out, self.fc3_out)
        self.fc3_bn = nn.BatchNorm1d(self.fc3_out)
        #         dropout_rate_fc3 = trial.suggest_categorical("Base1_RCM_triCNN_combined_9000_dropout_rate_fc3", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc3 = 0.4
        self.drop_fc3 = nn.Dropout(p=dropout_rate_fc3)

        #         self.fc4_out = trial.suggest_categorical('Base1_RCM_triCNN_combined_9000_fc4_out', [4, 8, 16])
        self.fc4_out = 4
        self.fc4 = nn.Linear(self.fc3_out, self.fc4_out)
        self.fc4_bn = nn.BatchNorm1d(self.fc4_out)
        dropout_rate_fc4 = 0.1
        #         dropout_rate_fc4 = trial.suggest_categorical("Base1_RCM_triCNN_combined_9000_dropout_rate_fc4", [0, 0.1, 0.2, 0.4])
        self.drop_fc4 = nn.Dropout(p=dropout_rate_fc4)

        self.fc5_out = 2
        self.fc5 = nn.Linear(self.fc4_out, self.fc5_out)

    def forward(self, seq_upper_lower_feature, rcm_flanking, rcm_upper, rcm_lower):
        x1 = self.cnn(seq_upper_lower_feature)
        ### layer to process junction seq
        x1 = self.fc1(x1)
        #         print(x_12.shape)
        x1 = self.drop_nn1(torch.relu(self.fc1_bn(x1)))
        x1 = self.fc2(x1)
        x1 = self.drop_nn2(torch.relu(self.fc2_bn(x1)))

        ### layer to process RCM information
        x2 = self.rcm_flanking(rcm_flanking)
        x3 = self.rcm_upper(rcm_upper)
        x4 = self.rcm_lower(rcm_lower)

        x_234 = torch.cat((x2, x3, x4), dim=1)
        x_234 = self.rcm_concate_fc1(x_234)

        x_234 = self.drop_rcm_concate_fc1(torch.relu(self.rcm_concate_fc1_bn(x_234)))

        x_234 = self.rcm_concate_fc2(x_234)
        x_234 = self.drop_rcm_concate_fc2(torch.relu(self.rcm_concate_fc2_bn(x_234)))

        x_1234 = F.normalize(torch.cat((x1, x_234), dim=1), dim=1)

        out = self.drop_fc3(torch.relu(self.fc3_bn(self.fc3(x_1234))))
        out = self.drop_fc4(torch.relu(self.fc4_bn(self.fc4(out))))

        out = self.fc5(out)

        return out


#### get the model structure to flatten layers first

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


class Base1_RCM_triCNN_combined_10000(nn.Module):

    def __init__(self, trial):
        super(Base1_RCM_triCNN_combined_10000, self).__init__()

        ### cnn for model1

        self.cnn = Model1_optuna_10000(trial)

        ### cnn for RCM features
        self.rcm_flanking = RCM_optuna_flanking_10000(trial)
        self.rcm_upper = RCM_optuna_upper_10000(trial)
        self.rcm_lower = RCM_optuna_lower_10000(trial)

        ### fc1 layers from the ConcatModel1_optuna_10000

        self.fc1_input_dim = self.cnn.conv2_out_dim * self.cnn.out_channel2

        self.fc1_out = 128

        self.fc1 = nn.Linear(self.fc1_input_dim, self.fc1_out)

        self.fc1_bn = nn.BatchNorm1d(self.fc1_out)

        dropout_rate_fc1 = 0
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc2 layers from the ConcatModel1_optuna_10000

        self.fc2_out = 4
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

        self.fc2_bn = nn.BatchNorm1d(self.fc2_out)

        dropout_rate_fc2 = 0
        self.drop_nn2 = nn.Dropout(p=dropout_rate_fc2)

        ### fc1 layers from the RCM_optuna_concate_10000
        self.rcm_concate_fc1_out = 512
        self.rcm_concate_fc1_in = self.rcm_flanking.conv2_out_dim * self.rcm_flanking.out_channel2 + \
                                  self.rcm_upper.conv2_out_dim * self.rcm_upper.out_channel2 + \
                                  self.rcm_lower.conv2_out_dim * self.rcm_lower.out_channel2

        self.rcm_concate_fc1 = nn.Linear(self.rcm_concate_fc1_in, self.rcm_concate_fc1_out)

        self.rcm_concate_fc1_bn = nn.BatchNorm1d(self.rcm_concate_fc1_out)

        dropout_rate_rcm_concate_fc1 = 0.4
        self.drop_rcm_concate_fc1 = nn.Dropout(p=dropout_rate_rcm_concate_fc1)

        ### fc2 layers from the RCM_optuna_concate_10000
        self.rcm_concate_fc2_out = 8
        self.rcm_concate_fc2 = nn.Linear(self.rcm_concate_fc1_out, self.rcm_concate_fc2_out)

        self.rcm_concate_fc2_bn = nn.BatchNorm1d(self.rcm_concate_fc2_out)

        dropout_rate_rcm_concate_fc2 = 0.2
        self.drop_rcm_concate_fc2 = nn.Dropout(p=dropout_rate_rcm_concate_fc2)

        ## newly added two layers

        #         self.fc3_out = trial.suggest_categorical('Base1_RCM_triCNN_combined_10000_fc3_out', [8, 16, 32])
        self.fc3_out = 8
        self.fc3 = nn.Linear(self.rcm_concate_fc2_out + self.fc2_out, self.fc3_out)
        self.fc3_bn = nn.BatchNorm1d(self.fc3_out)
        #         dropout_rate_fc3 = trial.suggest_categorical("Base1_RCM_triCNN_combined_10000_dropout_rate_fc3", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc3 = 0.4
        self.drop_fc3 = nn.Dropout(p=dropout_rate_fc3)

        #         self.fc4_out = trial.suggest_categorical('Base1_RCM_triCNN_combined_10000_fc4_out', [4, 8, 16])
        self.fc4_out = 4
        self.fc4 = nn.Linear(self.fc3_out, self.fc4_out)
        self.fc4_bn = nn.BatchNorm1d(self.fc4_out)
        dropout_rate_fc4 = 0.4
        #         dropout_rate_fc4 = trial.suggest_categorical("Base1_RCM_triCNN_combined_10000_dropout_rate_fc4", [0, 0.1, 0.2, 0.4])
        self.drop_fc4 = nn.Dropout(p=dropout_rate_fc4)

        self.fc5_out = 2
        self.fc5 = nn.Linear(self.fc4_out, self.fc5_out)

    def forward(self, seq_upper_lower_feature, rcm_flanking, rcm_upper, rcm_lower):
        x1 = self.cnn(seq_upper_lower_feature)
        ### layer to process junction seq
        x1 = self.fc1(x1)
        #         print(x_12.shape)
        x1 = self.drop_nn1(torch.relu(self.fc1_bn(x1)))
        x1 = self.fc2(x1)
        x1 = self.drop_nn2(torch.relu(self.fc2_bn(x1)))

        ### layer to process RCM information
        x2 = self.rcm_flanking(rcm_flanking)
        x3 = self.rcm_upper(rcm_upper)
        x4 = self.rcm_lower(rcm_lower)

        x_234 = torch.cat((x2, x3, x4), dim=1)
        x_234 = self.rcm_concate_fc1(x_234)

        x_234 = self.drop_rcm_concate_fc1(torch.relu(self.rcm_concate_fc1_bn(x_234)))

        x_234 = self.rcm_concate_fc2(x_234)
        x_234 = self.drop_rcm_concate_fc2(torch.relu(self.rcm_concate_fc2_bn(x_234)))

        x_1234 = F.normalize(torch.cat((x1, x_234), dim=1), dim=1)

        out = self.drop_fc3(torch.relu(self.fc3_bn(self.fc3(x_1234))))
        out = self.drop_fc4(torch.relu(self.fc4_bn(self.fc4(out))))

        out = self.fc5(out)

        return out


#### get the model structure to flatten layers first

class Model1_optuna_8000(nn.Module):
    '''
        This model take in input sequence 4 X 400 with 1 CNN layer
    '''

    def __init__(self, trial):
        super(Model1_optuna_8000, self).__init__()
        ### first CNN layer
        #         self.out_channel1 = trial.suggest_categorical('out_channel1', [128, 256, 512])
        self.out_channel1 = 512
        #         kernel_size1 = trial.suggest_categorical('kernel_size1', [13, 15, 17, 19, 21])
        kernel_size1 = 13
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel1, \
                               kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)
        #         self.maxpool1 = trial.suggest_categorical('maxpool1', [5, 10, 20])
        self.maxpool1 = 5
        self.conv1_out_dim = 400 // self.maxpool1

        #         self.out_channel2 = trial.suggest_categorical('out_channel2', [128, 256, 512])
        self.out_channel2 = 512
        #         kernel_size2 = trial.suggest_categorical('kernel_size2', [13, 15, 17, 19, 21])
        kernel_size2 = 13
        self.conv2 = nn.Conv1d(in_channels=self.out_channel1, out_channels=self.out_channel2, \
                               kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(self.out_channel2)
        #         self.maxpool2 = trial.suggest_categorical('maxpool2', [5, 10, 20])
        self.maxpool2 = 5
        self.conv2_out_dim = 400 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = x
        out = torch.relu(self.conv1_bn(self.conv1(out)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(-1, self.out_channel2 * self.conv2_out_dim)
        return out


class RCM_optuna_flanking_8000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the flanking introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_flanking_8000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('flanking_out_channel1', [128, 256, 512])
        self.out_channel1 = 512

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


class RCM_optuna_upper_8000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the upper introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_upper_8000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('upper_out_channel1', [128, 256, 512])
        self.out_channel1 = 128

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=self.out_channel1, \
                               kernel_size=5, stride=5, padding=0)

        self.conv1_bn = nn.BatchNorm1d(self.out_channel1)

        #         self.out_channel2 = trial.suggest_categorical('upper_out_channel2', [128, 256, 512])
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


class RCM_optuna_lower_8000(nn.Module):
    '''
        This is for 2-d model to process the RCM score distribution of the lower introns
    '''

    def __init__(self, trial):
        super(RCM_optuna_lower_8000, self).__init__()

        # convlayer 1
        #         self.out_channel1 = trial.suggest_categorical('lower_out_channel1', [128, 256, 512])
        self.out_channel1 = 128

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


class Base1_RCM_triCNN_combined_8000(nn.Module):

    def __init__(self, trial):
        super(Base1_RCM_triCNN_combined_8000, self).__init__()

        ### cnn for model1

        self.cnn = Model1_optuna_8000(trial)

        ### cnn for RCM features
        self.rcm_flanking = RCM_optuna_flanking_8000(trial)
        self.rcm_upper = RCM_optuna_upper_8000(trial)
        self.rcm_lower = RCM_optuna_lower_8000(trial)

        ### fc1 layers from the ConcatModel1_optuna_8000

        self.fc1_input_dim = self.cnn.conv2_out_dim * self.cnn.out_channel2

        self.fc1_out = 512

        self.fc1 = nn.Linear(self.fc1_input_dim, self.fc1_out)

        self.fc1_bn = nn.BatchNorm1d(self.fc1_out)

        dropout_rate_fc1 = 0
        self.drop_nn1 = nn.Dropout(p=dropout_rate_fc1)

        # fc2 layers from the ConcatModel1_optuna_8000

        self.fc2_out = 4
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

        self.fc2_bn = nn.BatchNorm1d(self.fc2_out)

        dropout_rate_fc2 = 0.2
        self.drop_nn2 = nn.Dropout(p=dropout_rate_fc2)

        ### fc1 layers from the RCM_optuna_concate_8000
        self.rcm_concate_fc1_out = 256
        self.rcm_concate_fc1_in = self.rcm_flanking.conv2_out_dim * self.rcm_flanking.out_channel2 + \
                                  self.rcm_upper.conv2_out_dim * self.rcm_upper.out_channel2 + \
                                  self.rcm_lower.conv2_out_dim * self.rcm_lower.out_channel2

        self.rcm_concate_fc1 = nn.Linear(self.rcm_concate_fc1_in, self.rcm_concate_fc1_out)

        self.rcm_concate_fc1_bn = nn.BatchNorm1d(self.rcm_concate_fc1_out)

        dropout_rate_rcm_concate_fc1 = 0.1
        self.drop_rcm_concate_fc1 = nn.Dropout(p=dropout_rate_rcm_concate_fc1)

        ### fc2 layers from the RCM_optuna_concate_8000
        self.rcm_concate_fc2_out = 16
        self.rcm_concate_fc2 = nn.Linear(self.rcm_concate_fc1_out, self.rcm_concate_fc2_out)

        self.rcm_concate_fc2_bn = nn.BatchNorm1d(self.rcm_concate_fc2_out)

        dropout_rate_rcm_concate_fc2 = 0
        self.drop_rcm_concate_fc2 = nn.Dropout(p=dropout_rate_rcm_concate_fc2)

        ## newly added two layers

        #         self.fc3_out = trial.suggest_categorical('Base1_RCM_triCNN_combined_8000_fc3_out', [8, 16, 32])
        self.fc3_out = 16
        self.fc3 = nn.Linear(self.rcm_concate_fc2_out + self.fc2_out, self.fc3_out)
        self.fc3_bn = nn.BatchNorm1d(self.fc3_out)
        #         dropout_rate_fc3 = trial.suggest_categorical("Base1_RCM_triCNN_combined_8000_dropout_rate_fc3", [0, 0.1, 0.2, 0.4])
        dropout_rate_fc3 = 0.1
        self.drop_fc3 = nn.Dropout(p=dropout_rate_fc3)

        #         self.fc4_out = trial.suggest_categorical('Base1_RCM_triCNN_combined_8000_fc4_out', [4, 8, 16])
        self.fc4_out = 4
        self.fc4 = nn.Linear(self.fc3_out, self.fc4_out)
        self.fc4_bn = nn.BatchNorm1d(self.fc4_out)
        dropout_rate_fc4 = 0
        #         dropout_rate_fc4 = trial.suggest_categorical("Base1_RCM_triCNN_combined_8000_dropout_rate_fc4", [0, 0.1, 0.2, 0.4])
        self.drop_fc4 = nn.Dropout(p=dropout_rate_fc4)

        self.fc5_out = 2
        self.fc5 = nn.Linear(self.fc4_out, self.fc5_out)

    def forward(self, seq_upper_lower_feature, rcm_flanking, rcm_upper, rcm_lower):
        x1 = self.cnn(seq_upper_lower_feature)
        ### layer to process junction seq
        x1 = self.fc1(x1)
        #         print(x_12.shape)
        x1 = self.drop_nn1(torch.relu(self.fc1_bn(x1)))
        x1 = self.fc2(x1)
        x1 = self.drop_nn2(torch.relu(self.fc2_bn(x1)))

        ### layer to process RCM information
        x2 = self.rcm_flanking(rcm_flanking)
        x3 = self.rcm_upper(rcm_upper)
        x4 = self.rcm_lower(rcm_lower)

        x_234 = torch.cat((x2, x3, x4), dim=1)
        x_234 = self.rcm_concate_fc1(x_234)

        x_234 = self.drop_rcm_concate_fc1(torch.relu(self.rcm_concate_fc1_bn(x_234)))

        x_234 = self.rcm_concate_fc2(x_234)
        x_234 = self.drop_rcm_concate_fc2(torch.relu(self.rcm_concate_fc2_bn(x_234)))

        x_1234 = F.normalize(torch.cat((x1, x_234), dim=1), dim=1)

        out = self.drop_fc3(torch.relu(self.fc3_bn(self.fc3(x_1234))))
        out = self.drop_fc4(torch.relu(self.fc4_bn(self.fc4(out))))

        out = self.fc5(out)

        return out








