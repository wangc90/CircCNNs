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
### import Dataset prepartion and model training classes Auxiliary_Codes folder
from BS_LS_DataSet_3 import BS_LS_DataSet_Prep, BS_LS_upper_lower_rcm, BS_LS_upper_lower_concat_rcm, RCM_Score
from pre_trained_model_structure import *


def Model_eva(model_path, test_data_set):
    '''
    ### load the saved best model and do the evaluation on the independent test dataset

    '''
    saved_model = torch.load(model_path).to('cuda')

    saved_model.eval()

    # test_data_set = torch.utils.data.Subset(BS_LS_test_dataset)

    data_loader = DataLoader(test_data_set, batch_size=100)

    with torch.no_grad():
        all_test_labels = []
        all_preds_prob = []

        correct, total = 0.0, 0.0
        for *test_features, test_labels in data_loader:
            #### change it to cuda:1 when evaluation for rcm models
            test_labels = test_labels.type(torch.LongTensor).to('cuda')
            test_features = [i.to('cuda') for i in test_features]

            preds = saved_model(*test_features)
            ## get the predited probability
            preds_prob = F.softmax(preds, dim=1)[:, 1]

            _, preds_labels = torch.max(preds, 1)

            correct += (preds_labels == test_labels).sum().item()
            total += test_labels.shape[0]

            all_test_labels.extend(test_labels.cpu().numpy().tolist())
            all_preds_prob.extend(preds_prob.cpu().numpy().tolist())

        val_acc = round(correct / total, 4)

        print(val_acc)

    return torch.from_numpy(np.array(all_test_labels)), torch.from_numpy(np.array(all_preds_prob))


def testing_set_base2(BS_LS_coordinates_path, hg19_seq_dict_json_path,
                      flanking_dict_folder):

    '''
    :return: testing set for base 2 models
    '''
    BS_LS_coordinates_path = BS_LS_coordinates_path
    hg19_seq_dict_json_path = hg19_seq_dict_json_path
    ## need to specify where to store the flanking sequence
    flanking_dict_folder = flanking_dict_folder
    bs_ls_dataset = BS_LS_DataSet_Prep(BS_LS_coordinates_path=BS_LS_coordinates_path,
                                       hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                                       flanking_dict_folder=flanking_dict_folder,
                                       flanking_junction_bps=100,
                                       flanking_intron_bps=5000,
                                       training_size=8000)

    ### generate the junction and flanking intron dict if not already exists
    # bs_ls_dataset.get_junction_flanking_intron_seq()

    ### get the testing keys
    _, _, test_keys = bs_ls_dataset.get_train_test_keys()

    train_torch_upper_features, train_torch_lower_features,\
    train_torch_labels = bs_ls_dataset.seq_to_tensor(data_keys=test_keys,
                                                        rcm_folder=None,
                                                        is_rcm=False,
                                                        is_upper_lower_concat=False)

    BS_LS_dataset_base2 = BS_LS_upper_lower_rcm(include_rcm=False,
                                                seq_upper_feature=train_torch_upper_features,
                                                seq_lower_feature=train_torch_lower_features,
                                                flanking_rcm=None,
                                                upper_rcm=None,
                                                lower_rcm=None,
                                                label=train_torch_labels)

    return BS_LS_dataset_base2


def testing_set_base1(BS_LS_coordinates_path, hg19_seq_dict_json_path,
                      flanking_dict_folder):

    '''
    :return: testing set for base 2 models
    '''
    BS_LS_coordinates_path = BS_LS_coordinates_path
    hg19_seq_dict_json_path = hg19_seq_dict_json_path
    ## need to specify where to store the flanking sequence
    flanking_dict_folder = flanking_dict_folder
    bs_ls_dataset = BS_LS_DataSet_Prep(BS_LS_coordinates_path=BS_LS_coordinates_path,
                                       hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                                       flanking_dict_folder=flanking_dict_folder,
                                       flanking_junction_bps=100,
                                       flanking_intron_bps=5000,
                                       training_size=8000)

    _, _, test_keys = bs_ls_dataset.get_train_test_keys()
    train_torch_upper_lower_features, train_torch_labels = bs_ls_dataset.seq_to_tensor(data_keys=test_keys,
                                                                                                rcm_folder=None,
                                                                                                is_rcm=False,
                                                                                                is_upper_lower_concat=True)

    BS_LS_dataset_base1 = BS_LS_upper_lower_concat_rcm(include_rcm=False,
                                                       seq_upper_lower_feature=train_torch_upper_lower_features,
                                                       flanking_rcm=None,
                                                       upper_rcm=None,
                                                       lower_rcm=None,
                                                       label=train_torch_labels)
    return BS_LS_dataset_base1


def testing_set_combined2(BS_LS_coordinates_path, hg19_seq_dict_json_path,
                          flanking_dict_folder, rcm_scores_folder):
    ## These need to be changed for redhawks
    BS_LS_coordinates_path = BS_LS_coordinates_path
    hg19_seq_dict_json_path = hg19_seq_dict_json_path
    ## need to specify where to store the flanking sequence
    flanking_dict_folder = flanking_dict_folder
    bs_ls_dataset = BS_LS_DataSet_Prep(BS_LS_coordinates_path=BS_LS_coordinates_path,
                                       hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                                       flanking_dict_folder=flanking_dict_folder,
                                       flanking_junction_bps=100,
                                       flanking_intron_bps=5000,
                                       training_size=8000)

    ## generate the junction and flanking intron dict
    # bs_ls_dataset.get_junction_flanking_intron_seq()

    _, _, test_keys = bs_ls_dataset.get_train_test_keys()

    rcm_scores_folder = rcm_scores_folder
    # try without rcm features
    train_torch_upper_features, train_torch_lower_features, train_torch_flanking_rcm, train_torch_upper_rcm, \
    train_torch_lower_rcm, train_torch_labels = bs_ls_dataset.seq_to_tensor(data_keys=test_keys,
                                                                            rcm_folder=rcm_scores_folder,
                                                                            is_rcm=True,
                                                                            is_upper_lower_concat=False)

    BS_LS_dataset_base2_combined = BS_LS_upper_lower_rcm(include_rcm=True,
                                                         seq_upper_feature=train_torch_upper_features,
                                                         seq_lower_feature=train_torch_lower_features,
                                                         flanking_rcm=train_torch_flanking_rcm,
                                                         upper_rcm=train_torch_upper_rcm,
                                                         lower_rcm=train_torch_lower_rcm,
                                                         label=train_torch_labels)

    return BS_LS_dataset_base2_combined



def testing_set_combined1(BS_LS_coordinates_path, hg19_seq_dict_json_path,
                          flanking_dict_folder, rcm_scores_folder):
    ## These need to be changed for redhawks
    BS_LS_coordinates_path = BS_LS_coordinates_path
    hg19_seq_dict_json_path = hg19_seq_dict_json_path
    ## need to specify where to store the flanking sequence
    flanking_dict_folder = flanking_dict_folder
    bs_ls_dataset = BS_LS_DataSet_Prep(BS_LS_coordinates_path=BS_LS_coordinates_path,
                                       hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                                       flanking_dict_folder=flanking_dict_folder,
                                       flanking_junction_bps=100,
                                       flanking_intron_bps=5000,
                                       training_size=8000)
    ## generate the junction and flanking intron dict
    # bs_ls_dataset.get_junction_flanking_intron_seq()

    _, _, test_keys = bs_ls_dataset.get_train_test_keys()

    rcm_scores_folder = rcm_scores_folder
    # try without rcm features
    train_torch_upper_lower_features, train_torch_flanking_rcm, train_torch_upper_rcm,\
    train_torch_lower_rcm, train_torch_labels = bs_ls_dataset.seq_to_tensor(data_keys=test_keys,
                                                                            rcm_folder=rcm_scores_folder,
                                                                            is_rcm=True,
                                                                            is_upper_lower_concat=True)

    BS_LS_dataset_base1_combined = BS_LS_upper_lower_concat_rcm(include_rcm=True,
                                          seq_upper_lower_feature=train_torch_upper_lower_features,
                                          flanking_rcm=train_torch_flanking_rcm,
                                          upper_rcm=train_torch_upper_rcm,
                                          lower_rcm=train_torch_lower_rcm,
                                          label=train_torch_labels)

    return BS_LS_dataset_base1_combined


def BS_LS_pred(model_choice=None):
    '''
    This function is used for BS or LS exon pairs prediction based on one of the 12
    retrained models:
    choice=1: base1_retraining_10000, choice=2: base1_retraining_9000, choice=3: base1_retraining_8000
    choice=4: base2_retraining_10000, choice=5: base2_retraining_9000, choice=6: base2_retraining_8000
    choice=7: combined1_10000, choice=8: combined1_9000, choice=9: combined1_8000
    choice=10: combined2_10000, choice=11: combined2_9000, choice=12: combined2_8000
    :return: print prediction accuracy and return predicted probability of being circRNA
    '''

    BS_LS_coordinates_path='/home/wangc90/circRNA/circRNA_Data/BS_LS_data/updated_data/BS_LS_coordinates_final.csv'
    hg19_seq_dict_json_path = '/home/wangc90/circRNA/circRNA_Data/hg19_seq/hg19_seq_dict.json'
    flanking_dict_folder = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/flanking_dicts/'
    rcm_scores_folder = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/flanking_dicts/rcm_scores/'

    if model_choice in [1, 2, 3]:

        print('Preparing testing set')
        BS_LS_dataset = testing_set_base1(BS_LS_coordinates_path=BS_LS_coordinates_path,
                                                hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                                                flanking_dict_folder=flanking_dict_folder)
        print('Testing set is prepared')
    elif model_choice in [4, 5, 6]:
        print('Preparing testing set')
        BS_LS_dataset = testing_set_base2(BS_LS_coordinates_path=BS_LS_coordinates_path,
                          hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                          flanking_dict_folder=flanking_dict_folder)
        print('Testing set is prepared')
    elif model_choice in [7, 8, 9]:
        print('Preparing testing set')
        BS_LS_dataset = testing_set_combined1(BS_LS_coordinates_path=BS_LS_coordinates_path,
                                                        hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                                                        flanking_dict_folder=flanking_dict_folder,
                                                        rcm_scores_folder=rcm_scores_folder)
        print('Testing set is prepared')
    elif model_choice in [10, 11, 12]:
        print('Preparing testing set')
        BS_LS_dataset = testing_set_combined2(BS_LS_coordinates_path=BS_LS_coordinates_path,
                          hg19_seq_dict_json_path=hg19_seq_dict_json_path,
                          flanking_dict_folder=flanking_dict_folder,
                          rcm_scores_folder=rcm_scores_folder)
        print('Testing set is prepared')
    else:
        print('Model choice is invalid and should be between 1 to 12 integer')
        exit()

    print('Bringing the corresponding retrained model weight for circRNA prediction')

    if model_choice == 1:
        base_model1_10000_path = "../Trained_Model_Weights/Base_model1_retraining_10000/retrained_model_149.pt"
        base_model1_10000_test_labels, base_model1_10000_preds_prob = Model_eva(model_path=base_model1_10000_path, test_data_set=BS_LS_dataset)
        return base_model1_10000_preds_prob

    elif model_choice == 2:
        base_model1_9000_path = "../Trained_Model_Weights/Base_model1_retraining_9000/retrained_model_149.pt"
        base_model1_9000_test_labels, base_model1_9000_preds_prob = Model_eva(model_path=base_model1_9000_path, \
                                                                              test_data_set=BS_LS_dataset)
        return base_model1_9000_preds_prob


base_model2_8000_path = "../Trained_Model_Weights/Base_model2_retraining_8000/retrained_model_119.pt"
base_model2_8000_test_labels, base_model2_8000_preds_prob = Model_eva(model_path=base_model2_8000_path,\
                                                  test_data_set=BS_LS_dataset_base2)

base_model2_9000_path = "../Trained_Model_Weights/Base_model2_retraining_9000/retrained_model_89.pt"
base_model2_9000_test_labels, base_model2_9000_preds_prob = Model_eva(model_path=base_model2_9000_path,\
                                                  test_data_set=BS_LS_dataset_base2)

base_model2_10000_path = "../Trained_Model_Weights/Base_model2_retraining_10000/retrained_model_149.pt"
base_model2_10000_test_labels, base_model2_10000_preds_prob = Model_eva(model_path=base_model2_10000_path,\
                                                  test_data_set=BS_LS_dataset_base2)

### bring in the model weights for optimized base model 1 on training set1, 2 and 3
base_model1_8000_path = "../Trained_Model_Weights/Base_model1_retraining_8000/retrained_model_149.pt"
base_model1_8000_test_labels, base_model1_8000_preds_prob = Model_eva(model_path=base_model1_8000_path,\
                                                  test_data_set=BS_LS_dataset_base1)






### bring in the model weights for base2 combined model on combining set1, 2 and 3
base_model2_combined_8000_path = '../Trained_Model_Weights/Combined_model2_retraining_8000/retrained_model_89.pt'

base_model2_combined_8000_test_labels, base_model2_combined_8000_preds_prob = Model_eva(model_path=base_model2_combined_8000_path,
                                                                    test_data_set=BS_LS_dataset_combined2)

base_model2_combined_9000_path = '../Trained_Model_Weights/Combined_model2_retraining_9000/retrained_model_59.pt'

base_model2_combined_9000_test_labels, base_model2_combined_9000_preds_prob = Model_eva(model_path=base_model2_combined_9000_path,
                                                                    test_data_set=BS_LS_dataset_combined2)

base_model2_combined_10000_path = '../Trained_Model_Weights/Combined_model2_retraining_10000/retrained_model_89.pt'

base_model2_combined_10000_test_labels, base_model2_combined_10000_preds_prob = Model_eva(model_path=base_model2_combined_10000_path,
                                                                    test_data_set=BS_LS_dataset_combined2)

### bring in the model weights for base1 combined model on combining set1, 2 and 3

base_model1_combined_8000_path = '../Trained_Model_Weights/Combined_model1_retraining_8000/retrained_model_59.pt'

base_model1_combined_8000_test_labels, base_model1_combined_8000_preds_prob = Model_eva(model_path=base_model1_combined_8000_path,
                                                                    test_data_set=BS_LS_dataset_combined1)


base_model1_combined_9000_path = '../Trained_Model_Weights/Combined_model1_retraining_9000/retrained_model_119.pt'

base_model1_combined_9000_test_labels, base_model1_combined_9000_preds_prob = Model_eva(model_path=base_model1_combined_9000_path,
                                                                    test_data_set=BS_LS_dataset_combined1)

base_model1_combined_10000_path = '../Trained_Model_Weights/Combined_model1_retraining_10000/retrained_model_149.pt'

base_model1_combined_10000_test_labels, base_model1_combined_10000_preds_prob = Model_eva(model_path=base_model1_combined_10000_path,
                                                                    test_data_set=BS_LS_dataset_combined1)