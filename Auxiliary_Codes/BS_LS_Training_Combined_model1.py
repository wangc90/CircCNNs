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

### training for the Combined model1: several options in the ObjectCV function is removed

def Objective(device, trial, fold, model, optimizer,
              patience, epochs, criterion, train_loader,
              val_loader, model_folder):
    #     print(f"I'm in the fold: {fold}")
    #     print('here is my model structure', model)
    ### implement the early stopping based on the validation loss change
    last_val_loss = 1000  # set to some big number
    counter = 0  # count the patience so far

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
            ## first evaluate the training acc
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

            #             print(f"I'am finished the epoch {epoch} evaluation on the training set")

            if (epoch + 1) % 50 == 0:
                print(f'fold {fold + 1}, epoch {epoch + 1}, training loss {running_loss}, train accuracy {train_acc}')

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

            #             print(f"I'am finished the epoch {epoch} evaluation on the validation set")

            total_val_loss = np.sum(val_loss_list)

            val_labels_total = torch.cat(val_labels_list, dim=0)
            val_preds_total = torch.cat(val_preds_list, dim=0)
            ## calculate the f1 score
            #                     f1_score = f1(val_preds_total, val_labels_total)

            if (epoch + 1) % 50 == 0:
                print(f'fold {fold + 1}, epoch {epoch + 1}, val accuracy {val_acc}')

        ### early stopping checking and save the model after each epoch if the trial is not pruned
        if total_val_loss <= last_val_loss:
            #             print(f"the total val loss is {total_val_loss} on epoch {epoch}")
            last_val_loss = total_val_loss
            best_val_acc = val_acc  # save the best val_acc so far
            #                 print(f'epoch {epoch+1} total val loss:{total_val_loss}')

            #                 print(f'epoch {epoch + 1}, val loss {total_val_loss}, val accuracy {val_acc}')
            ## set counter to 0 to start checking if the next 10 consecutive epoches are having reduced val loss
            counter = 0


            ## this line will overwrite the model and save the best one in each fold
            ## with the lowest val loss in each trial
            model_path = f"{model_folder}/fold{fold + 1}_trial{trial.number}.pt"
            torch.save(model, model_path)

        else:
            #                 print(f'epoch {epoch+1} total val loss:{total_val_loss}')
            counter += 1
            #                 print(counter)
            if counter >= patience:
                break  # break out of the epoch loop and into the next fold

    return best_val_acc  ## best validation accuracy in each fold


class Objective_CV:

    def __init__(self, patience, cv, model, dataset,
                 val_acc_folder, model_folder):

        self.patience = patience  ## number of epochs for early stopping the model training
        self.cv = cv  ## number of CV
        self.model = model  ## pass the corresponding model
        self.dataset = dataset  ## the corresponding dataset object
        self.val_acc_folder = val_acc_folder  ## folder to store the cross_validation accuracy
        self.model_folder = model_folder  ## folder to store the trained model for later testing dataset evaluation

    def __call__(self, trial):

        ### just use the sequence feature for now
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        # lr = 0.00015
        l2_lambda = trial.suggest_float("l2_lambda", 1e-8, 1e-3, log=True)
        # l2_lambda = 1.5e-6

        ### fix and use the maximal allowed batch size
        #         batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        batch_size = 512

        ### optimize epoch number
        epochs = 150

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

            ### for the model the process the concatenated upper and lower the is_rcm is always False
            model = self.model(trial).to(device=device)
            # print(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)

            accuracy = Objective(device, trial, fold=fold, model=model, optimizer=optimizer,
                                 patience=self.patience, epochs=epochs, criterion=criterion,
                                 train_loader=train_loader, val_loader=val_loader,
                                 model_folder=self.model_folder)

            val_acc_list.append(accuracy)

        ### to speed up training for cv just maximize the val_acc from the 3 fold cv
        ## choose the best model structure and hyperparameters based on average 3-cv validation acc

        avg_acc_val = np.mean(val_acc_list)

        val_acc_path = f"{self.val_acc_folder}/val_acc.csv"

        val_acc_str = '\t'.join([str(i) for i in val_acc_list])
        with open(val_acc_path, 'a') as f:
            f.write('trial' + str(trial.number) + '\t' + val_acc_str + '\n')

        return avg_acc_val
