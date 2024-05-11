import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
from pre_trained_model_structure import *

###Give a list of testing coordinate, use the trained models for prediction;
# for example a csv file like this, we want to make predictions on if these exon pairs are going to form circRNAs or not

##chr19|58921331|58929694|+
##chr9|91656940|91660748|-
##chr19|5724818|5768253|+


def reverse_complement(input_seq):
    '''
    This function take a sequence and returns the reverse complementary sequence
    '''
    complement_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'N': 'N'}
    out_seq = ''.join([complement_dict[base] for base in list(input_seq)[::-1]])
    return out_seq

def seq_to_matrix(input_seq):
    '''
        This function takes a DNA sequence and return a one-hot encoded matrix of 4 X N (length of input_seq)
    '''
    row_index = {'A': 0, 'G': 1, 'C': 2, 'T': 3}  # should exclude the 'Ns' in the input sequence

    # initialize the 4 X N 0 matrix:
    input_mat = np.zeros((4, len(input_seq)))

    for col_index, base in enumerate(input_seq):
        input_mat[row_index[base]][col_index] = 1
    return input_mat


### Prepare the testing dataset

def get_junction_seq(testing_df_path):
    '''
        return the junction seq for a given testing key
    '''
    junction_seq = {}

    with open('/home/wangc90/circRNA/circRNA_Data/hg19_seq/hg19_seq_dict.json') as f:

        hg19_seq_dict = json.load(f)

        testing_coordinates_df = pd.read_csv(testing_df_path, sep=',', header=None)
        for _, row in testing_coordinates_df.itertuples():
            line_contents = row.split('|')
            chrom = line_contents[0]
            strand = line_contents[-1]
            start = int(line_contents[1])
            end = int(line_contents[2])

            # get the corresponding chromosome DNA sequence
            dna_seq = hg19_seq_dict[chrom]

            # extract the spliced genomic sequence assuming the positive strand
            spliced_seq_P = dna_seq[start: end].upper()

            # extract the upper_intron junction seq assuming the positive strand
            upper_intron_P = dna_seq[start - 100: start].upper()

            # extract the lower_intron junction seq assuming the positive strand
            lower_intron_P = dna_seq[end: end + 100].upper()

            # skip the rows that have 'Ns' in the upper_intron or lower_intron
            if 'N' in upper_intron_P or 'N' in lower_intron_P:
                print(f'{row} has N in the extracted junctions')
                continue

            if strand == '-':
                ### get the junction sequence
                spliced_seq_N = reverse_complement(spliced_seq_P)
                upper_exon_N = spliced_seq_N[:100]
                lower_exon_N = spliced_seq_N[-100:]

                upper_intron_N = reverse_complement(lower_intron_P)
                lower_intron_N = reverse_complement(upper_intron_P)

                junction_seq[row] = {'upper_intron_' + "100": upper_intron_N,
                                     'upper_exon_' + "100": upper_exon_N,
                                     'lower_exon_' + "100": lower_exon_N,
                                     'lower_intron_' + "100": lower_intron_N}
            else:
                upper_exon_P = spliced_seq_P[:100]
                lower_exon_P = spliced_seq_P[-100:]

                junction_seq[row] = {'upper_intron_' + "100": upper_intron_P,
                                     'upper_exon_' + "100": upper_exon_P,
                                     'lower_exon_' + "100": lower_exon_P,
                                     'lower_intron_' + "100": lower_intron_P}

    return junction_seq


def seq_to_tensor(testing_df_path, is_upper_lower_concat=None):
    ### list to store the concatenated upper and lower sequence one-hot encoding
    if is_upper_lower_concat:
        all_torch_feature_list = []

    ### list to store the upper and lower sequence one-hot encoding separately
    else:
        all_torch_upper_feature_list = []
        all_torch_lower_feature_list = []

    junction_seq_dict = get_junction_seq(testing_df_path)
    for key in junction_seq_dict:
        value = junction_seq_dict[key]

        # concatenate the upper seq together and lower seq together for two separate CNN to process
        concatenated_upper_seq = value['upper_intron_100'] + \
                                 value['upper_exon_100']
        concatenated_lower_seq = value['lower_exon_100'] + \
                                 value['lower_intron_100']

        ### test whether want to concatenate the upper seq and lower seq together
        if is_upper_lower_concat:

            upper_lower_concat_seq = concatenated_upper_seq + concatenated_lower_seq
            upper_lower_concat_mat = seq_to_matrix(upper_lower_concat_seq)
            individual_upper_lower_torch = torch.from_numpy(upper_lower_concat_mat).to(torch.float32)
            all_torch_feature_list.append(individual_upper_lower_torch)

        else:
            individual_upper_mat = seq_to_matrix(concatenated_upper_seq)
            individual_lower_mat = seq_to_matrix(concatenated_lower_seq)

            # convert individual instance to torch
            individual_upper_torch = torch.from_numpy(individual_upper_mat).to(torch.float32)
            individual_lower_torch = torch.from_numpy(individual_lower_mat).to(torch.float32)

            all_torch_upper_feature_list.append(individual_upper_torch)
            all_torch_lower_feature_list.append(individual_lower_torch)

    if is_upper_lower_concat:

        all_torch_feature = torch.stack(all_torch_feature_list, dim=0)
        return all_torch_feature

    else:
        all_torch_upper_feature = torch.stack(all_torch_upper_feature_list, dim=0)
        all_torch_lower_feature = torch.stack(all_torch_lower_feature_list, dim=0)
        return all_torch_upper_feature, all_torch_lower_feature

### Dataset preparation for Basemodel1
class BS_LS_upper_lower_concat(Dataset):
    def __init__(self, seq_upper_lower_feature):
        # construction of the map-style datasets
        # data loading

        self.x1 = seq_upper_lower_feature

        self.n_samples = seq_upper_lower_feature.size()[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x1[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples

class BS_LS_upper_lower(Dataset):
    def __init__(self, seq_upper_feature, seq_lower_feature):
        # construction of the map-style datasets
        # data loading
        self.x1 = seq_upper_feature
        self.x2 = seq_lower_feature
        self.n_samples = seq_upper_feature.size()[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x1[index], self.x2[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


def BS_LS_pred(dataset, model_path, model_type):
    saved_model = torch.load(model_path).to('cuda')
    saved_model.eval()
    data_loader = DataLoader(dataset, batch_size=100)

    with torch.no_grad():

        all_preds_labels = []
        all_preds_prob = []

        for test_features in data_loader:
            if model_type == 2:
                test_features_ = [i.to('cuda') for i in test_features]
                preds = saved_model(*test_features_)
            else:
                preds = saved_model(test_features.to('cuda'))

            ## get the predited probability
            preds_prob = F.softmax(preds, dim=1)[:, 1]

            _, preds_labels = torch.max(preds, 1)

            all_preds_labels.extend(preds_labels.cpu().numpy().tolist())
            all_preds_prob.extend(preds_prob.cpu().numpy().tolist())

    return np.array(all_preds_labels), np.array(all_preds_prob)
