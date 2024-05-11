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
from BS_LS_DataSet_3 import BS_LS_DataSet_Prep, BS_LS_upper_lower_rcm
# from BS_LS_Training_Base_models import Objective, Objective_CV
from pre_trained_model_structure import *


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

### bring in the model weights for optimized base model 2:
base_model2_path = "/home/wangc90/circRNA/circRNA_Data/model_outputs/Base_model2_retraining/Base_model2_retraining_10000/retrained_model_149.pt"
base_model2 = torch.load(base_model2_path)

### bring in the model weights for optimized RCM_triCNN_all:
RCM_TriCNN_model_path = "/home/wangc90/circRNA/circRNA_Data/model_outputs/RCM_triCNN_retraining/RCM_triCNN_retraining_10000/retrained_model_149.pt"
RCM_TriCNN_model = torch.load(RCM_TriCNN_model_path)


def retraining(model, dataset, model_folder):
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    batch_size = 128

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #     print(len(train_loader))

    criterion = nn.CrossEntropyLoss()

    model = model('trial')
    #     print(model)

    pretrained_dict_upper_lower_concate_cnns = {k: v for k, v in base_model2.state_dict().items() if
                                                not 'concate_final' in k}

    pretrained_dict_rcm_cnns = {k: v for k, v in RCM_TriCNN_model.state_dict().items() if not 'fc3' in k}

    model_dict = model.state_dict()
    #             print('This is the original model weights')
    #             print(model_dict)
    model_dict.update(pretrained_dict_upper_lower_concate_cnns)
    model_dict.update(pretrained_dict_rcm_cnns)

    print('Loading the trained model weights to the Base2_RCM_triCNN_combined_10000 model')

    model.load_state_dict(model_dict)
    ### freeze these parameters
    model.cnn_upper.requires_grad_(False)
    model.cnn_lower.requires_grad_(False)

    model.rcm_flanking.requires_grad_(False)
    model.rcm_upper.requires_grad_(False)
    model.rcm_lower.requires_grad_(False)

    model.upper_lower_concate_fc1.requires_grad_(False)
    model.upper_lower_concate_fc1_bn.requires_grad_(False)

    model.upper_lower_concate_fc2.requires_grad_(False)
    model.upper_lower_concate_fc2_bn.requires_grad_(False)
    #             model.upper_lower_concate_final.requires_grad_(False)

    model.rcm_concate_fc1.requires_grad_(False)
    model.rcm_concate_fc1_bn.requires_grad_(False)

    model.rcm_concate_fc2.requires_grad_(False)
    model.rcm_concate_fc2_bn.requires_grad_(False)
    #             model.rcm_concate_final.requires_grad_(False)

    model.to(device=device)

    optimizer_name = 'Adam'
    lr = 0.000598
    l2_lambda = 4.611153e-09
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=l2_lambda)

    epochs = 90  ### reduce the epochs from 150 to 100 to reduce the potential overfitting
    #     epochs = 140
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


def Base2_RCM_triCNN_combined_10000_retraining():
    ### where to save the 3-fold CV validation acc

    ### where to save the retrained model
    model_folder = '/home/wangc90/circRNA/circRNA_Data/model_outputs/Combined_model2_retraining/Combined_model2_retraining_10000'

    ## These need to be changed for redhawks
    BS_LS_coordinates_path = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/updated_data/BS_LS_coordinates_final.csv'
    hg19_seq_dict_json_path = '/home/wangc90/circRNA/circRNA_Data/hg19_seq/hg19_seq_dict.json'
    flanking_dict_folder = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/flanking_dicts/'
    bs_ls_dataset = BS_LS_DataSet_Prep(BS_LS_coordinates_path=BS_LS_coordinates_path,
                                       hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                                       flanking_dict_folder=flanking_dict_folder,
                                       flanking_junction_bps=100,
                                       flanking_intron_bps=5000,
                                       training_size=10000)

    ## generate the junction and flanking intron dict
    bs_ls_dataset.get_junction_flanking_intron_seq()

    _, train_key2, test_keys = bs_ls_dataset.get_train_test_keys()

    rcm_scores_folder = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/flanking_dicts/rcm_scores/'
    # try without rcm features
    train_torch_upper_features, train_torch_lower_features, train_torch_flanking_rcm, train_torch_upper_rcm, \
    train_torch_lower_rcm, train_torch_labels = bs_ls_dataset.seq_to_tensor(data_keys=train_key2,
                                                                            rcm_folder=rcm_scores_folder,
                                                                            is_rcm=True,
                                                                            is_upper_lower_concat=False)

    BS_LS_dataset = BS_LS_upper_lower_rcm(include_rcm=True,
                                          seq_upper_feature=train_torch_upper_features,
                                          seq_lower_feature=train_torch_lower_features,
                                          flanking_rcm=train_torch_flanking_rcm,
                                          upper_rcm=train_torch_upper_rcm,
                                          lower_rcm=train_torch_lower_rcm,
                                          label=train_torch_labels)

    print(len(BS_LS_dataset))

    retraining(model=Base2_RCM_triCNN_combined_10000, dataset=BS_LS_dataset, model_folder=model_folder)

Base2_RCM_triCNN_combined_10000_retraining()

