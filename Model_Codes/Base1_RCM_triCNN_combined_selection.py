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
from BS_LS_DataSet_3 import BS_LS_DataSet_Prep, BS_LS_upper_lower_concat_rcm
# from BS_LS_Training_Base_models import Objective, Objective_CV
from pre_trained_model_structure import *


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

        self.fc3_out = trial.suggest_categorical('Base1_RCM_triCNN_combined_10000_fc3_out', [8, 16, 32])
        #         self.fc3_out = 32
        self.fc3 = nn.Linear(self.rcm_concate_fc2_out + self.fc2_out, self.fc3_out)
        self.fc3_bn = nn.BatchNorm1d(self.fc3_out)
        dropout_rate_fc3 = trial.suggest_categorical("Base1_RCM_triCNN_combined_10000_dropout_rate_fc3",
                                                     [0, 0.1, 0.2, 0.4])
        #         dropout_rate_fc3 = 0
        self.drop_fc3 = nn.Dropout(p=dropout_rate_fc3)

        self.fc4_out = trial.suggest_categorical('Base1_RCM_triCNN_combined_10000_fc4_out', [4, 8, 16])
        #         self.fc4_out = 6
        self.fc4 = nn.Linear(self.fc3_out, self.fc4_out)
        self.fc4_bn = nn.BatchNorm1d(self.fc4_out)
        #         dropout_rate_fc4 = 0
        dropout_rate_fc4 = trial.suggest_categorical("Base1_RCM_triCNN_combined_10000_dropout_rate_fc4",
                                                     [0, 0.1, 0.2, 0.4])
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

### bring in the model weights for optimized base model 1:
base_model1_path = "/home/wangc90/circRNA/circRNA_Data/model_outputs/Base_model1_retraining/Base_model1_retraining_10000/retrained_model_149.pt"
base_model1 = torch.load(base_model1_path)

### bring in the model weights for optimized RCM_triCNN_all:
RCM_TriCNN_model_path = "/home/wangc90/circRNA/circRNA_Data/model_outputs/RCM_triCNN_retraining/RCM_triCNN_retraining_10000/retrained_model_149.pt"
RCM_TriCNN_model = torch.load(RCM_TriCNN_model_path)


# ## Fix the pretrained model weights all the way to FC layer2 for both base model and RCM_triCNN model

def Objective(device, trial, fold, model, optimizer,
              epochs, criterion, train_loader,
              val_loader):
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

            correct, total = 0.0, 0.0
            val_loss_list = []

            for *val_features, val_labels in val_loader:
                ### this type conversion is just used for nn.CrossEntropy loss
                ### otherwise can be safely removed
                val_labels = val_labels.type(torch.LongTensor)

                val_labels = val_labels.to(device)
                val_features = [i.to(device) for i in val_features]

                # get the predition with the model parameters updated after each epoch
                val_preds = model(*val_features)

                # get the validation loss for the early stopping
                val_loss = criterion(val_preds, val_labels)
                val_loss_list.append(val_loss.item())

                # prediction for the nn.CrossEntropy loss
                _, preds_labels = torch.max(val_preds, 1)

                correct += (preds_labels == val_labels).sum().item()
                total += val_labels.shape[0]

            val_acc = round(correct / total, 4)
            total_val_loss = np.sum(val_loss_list)

        if (epoch + 1) % 20 == 0:
            print(f'fold {fold + 1}, epoch {epoch + 1}, val loss {total_val_loss} val accuracy {val_acc}')

    ### return the val_acc for at the end of the given epochs
    return val_acc


class Objective_CV:

    def __init__(self, cv, model, dataset,
                 val_acc_folder):

        self.cv = cv  ## number of CV
        self.model = model  ## pass the corresponding model
        self.dataset = dataset  ## the corresponding dataset object
        self.val_acc_folder = val_acc_folder  ## folder to store the cross_validation accuracy

    def __call__(self, trial):

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        ### lr, l2, batch_size and epochs need to be optimized for the combined models
        ### can use the optimal hyperparameters from the base and rcm models to narrow the range

        ## base1 lr=0.00016663145779864379, rcm lr=0.000016
        ## base1 l2=4.828346469350033e-07,rcm l2=3.358313e-08
        ## base1 batch_size=32, rcm batch_size=256
        ## base1 epochs=150, rcm epochs=150

        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        l2_lambda = trial.suggest_float("l2_lambda", 1e-9, 1e-6, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        epochs = trial.suggest_categorical("epochs", [30, 60, 90, 120, 150])

        optimizer_name = "Adam"

        criterion = nn.CrossEntropyLoss()

        kfold = KFold(n_splits=self.cv, shuffle=True)

        val_acc_list = []

        for fold, (train_index, val_index) in enumerate(kfold.split(np.arange(len(self.dataset)))):

            ### get the train and val loader
            train_subset = torch.utils.data.Subset(self.dataset, train_index)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)

            val_subset = torch.utils.data.Subset(self.dataset, val_index)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=4)

            ## model should be initilized here for each fold to have a new model with same hyperparameters
            model = self.model(trial)  # .to(device=device)
            #             print(model)
            ## load the pretrained model weights from base and rcm modesl

            pretrained_dict_upper_lower_concate_cnns = {k: v for k, v in base_model1.state_dict().items() if
                                                        not 'fc3' in k}

            pretrained_dict_rcm_cnns = {k: v for k, v in RCM_TriCNN_model.state_dict().items() if not 'fc3' in k}

            model_dict = model.state_dict()
            #             print('This is the original model weights')
            #             print(model_dict)
            model_dict.update(pretrained_dict_upper_lower_concate_cnns)
            model_dict.update(pretrained_dict_rcm_cnns)

            print('Loading the trained model weights to the Base1_RCM_triCNN_combined_10000 model')

            model.load_state_dict(model_dict)
            ### freeze these parameters
            model.cnn.requires_grad_(False)

            model.rcm_flanking.requires_grad_(False)
            model.rcm_upper.requires_grad_(False)
            model.rcm_lower.requires_grad_(False)

            model.fc1.requires_grad_(False)
            model.fc1_bn.requires_grad_(False)

            model.fc2.requires_grad_(False)
            model.fc2_bn.requires_grad_(False)
            #             model.upper_lower_concate_final.requires_grad_(False)

            model.rcm_concate_fc1.requires_grad_(False)
            model.rcm_concate_fc1_bn.requires_grad_(False)

            model.rcm_concate_fc2.requires_grad_(False)
            model.rcm_concate_fc2_bn.requires_grad_(False)
            #             model.rcm_concate_final.requires_grad_(False)

            model.to(device=device)

            #             for name, param in model.named_parameters():
            #                 print(name,param.requires_grad)

            #             print('This is the new model weights')
            #             print(model_dict)
            optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=l2_lambda)

            accuracy = Objective(device, trial, fold=fold, model=model, optimizer=optimizer,
                                 epochs=epochs, criterion=criterion,
                                 train_loader=train_loader, val_loader=val_loader)

            ## check for pruning
            trial.report(accuracy, fold)

            val_acc_list.append(accuracy)

            if trial.should_prune():
                raise optuna.TrialPruned()

        ### to speed up training for cv just maximize the val_acc from the 3 fold cv
        ## choose the best model structure and hyperparameters based on average 3-cv validation acc

        avg_acc_val = np.mean(val_acc_list)

        val_acc_path = f"{self.val_acc_folder}/val_acc.csv"

        val_acc_str = '\t'.join([str(i) for i in val_acc_list])
        with open(val_acc_path, 'a') as f:
            f.write('trial' + str(trial.number) + '\t' + val_acc_str + '\n')

        return avg_acc_val


def Base1_RCM_triCNN_combined_10000_optuna(num_trial):
    ### where to save the 3-fold CV validation acc

    val_acc_folder = '/home/wangc90/circRNA/circRNA_Data/model_outputs/Combined_models/10000/Base1_RCM_triCNN_combined_10000/val_acc_cv3'
    ### wehre to save the detailed optuna results
    optuna_folder = '/home/wangc90/circRNA/circRNA_Data/model_outputs/Combined_models/10000/Base1_RCM_triCNN_combined_10000/optuna'

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
    train_torch_upper_lower_features, train_torch_flanking_rcm, train_torch_upper_rcm, \
    train_torch_lower_rcm, train_torch_labels = bs_ls_dataset.seq_to_tensor(data_keys=train_key2,
                                                                            rcm_folder=rcm_scores_folder,
                                                                            is_rcm=True,
                                                                            is_upper_lower_concat=True)

    BS_LS_dataset = BS_LS_upper_lower_concat_rcm(include_rcm=True,
                                                 seq_upper_lower_feature=train_torch_upper_lower_features,
                                                 flanking_rcm=train_torch_flanking_rcm,
                                                 upper_rcm=train_torch_upper_rcm,
                                                 lower_rcm=train_torch_lower_rcm,
                                                 label=train_torch_labels)

    print(len(BS_LS_dataset))
    #     print(BS_LS_dataset[0][0].shape, BS_LS_dataset[0][1])

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                                direction='maximize')

    study.optimize(Objective_CV(cv=3,
                                model=Base1_RCM_triCNN_combined_10000,
                                dataset=BS_LS_dataset,
                                val_acc_folder=val_acc_folder), n_trials=num_trial, gc_after_trial=True)

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

Base1_RCM_triCNN_combined_10000_optuna(500)

