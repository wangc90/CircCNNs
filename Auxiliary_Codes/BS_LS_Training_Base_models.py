import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
import torch.optim as optim
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
# from torchmetrics.classification import F1Score
import pickle


def Objective(device, trial, fold, model, optimizer,
              epochs, criterion, train_loader,
              val_loader):
    ## lr decay: every 10 or 20 etc. epochs lr is halfed
    # step_size = trial.suggest_categorical("lr_decay_step", [10, 20, 30, 40, 50])
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    for epoch in range(epochs):
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

        ## start model validation
        model.eval()
        with torch.no_grad():
            # evaluate the validation accuracy and other metrics after each epoch
            correct, total = 0.0, 0.0
            val_loss_list = []
            # store the validation prediction and validation label for different metric calculation
            val_labels_list = []
            val_preds_list = []

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

                ## append the true and predict value in lists for the calculation of
                ## model performance
                val_preds_list.append(preds_labels)
                val_labels_list.append(val_labels)

                correct += (preds_labels == val_labels).sum().item()
                total += val_labels.shape[0]

            val_acc = round(correct / total, 4)
            total_val_loss = np.sum(val_loss_list)

        if (epoch + 1) % 20 == 0:
            print(f'fold {fold + 1}, epoch {epoch + 1}, val loss {total_val_loss} val accuracy {val_acc}')

        # scheduler.step()
    ### return the val_acc for at the end of the given epochs
    return val_acc


class Objective_CV:

    def __init__(self, cv, model, dataset, val_acc_folder):
        self.cv = cv  ## number of CV
        self.model = model  ## pass the corresponding model
        self.dataset = dataset  ## the corresponding dataset object
        self.val_acc_folder = val_acc_folder  ## folder to store the cross_validation accuracy

    def __call__(self, trial):
        device = torch.device('cuda:0') if torch. cuda.is_available() else torch.device('cpu')
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        # lr = 0.00015
        l2_lambda = trial.suggest_float("l2_lambda", 1e-9, 1e-6, log=True)
        # l2_lambda = 1.5e-6
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        # batch_size = 512
        optimizer_name = 'Adam'
        ### optimize epoch number
        epochs = trial.suggest_categorical("epochs", [30, 60, 90, 120, 150])
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

            model = self.model(trial).to(device=device)
            optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=l2_lambda)
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)

            accuracy = Objective(device, trial, fold=fold, model=model, optimizer=optimizer,
                                 epochs=epochs, criterion=criterion,
                                 train_loader=train_loader, val_loader=val_loader)

            ## record the val_acc for the current fold, prune the model based on
            ## the val_acc for the current fold
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
