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
# from torchmetrics.classification import F1Score
import pickle


class BS_LS_DataSet_Prep():
    def __init__(self, BS_LS_coordinates_path, hg19_seq_dict_json_path,
                 flanking_dict_folder,
                 flanking_junction_bps=None,
                 flanking_intron_bps=None,
                 training_size=None):

        self.BS_LS_coordinates_path = BS_LS_coordinates_path
        self.flanking_junction_bps = flanking_junction_bps
        self.flanking_intron_bps = flanking_intron_bps
        self.flanking_dict_folder = flanking_dict_folder
        ### bring in the hg19 genomic sequences
        with open(hg19_seq_dict_json_path) as f:
            self.hg19_seq_dict = json.load(f)
        self.junction_seq_json_name = f'BS_LS_junction_seq_{self.flanking_junction_bps}_bps'
        self.flanking_seq_json_name = f'BS_LS_intronic_flanking_seq_{self.flanking_intron_bps}_bps'
        self.training_size = training_size

    def reverse_complement(self, input_seq):
        '''
        This function take a sequence and returns the reverse complementary sequence
        '''
        complement_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'N': 'N'}
        out_seq = ''.join([complement_dict[base] for base in list(input_seq)[::-1]])
        return out_seq

    def get_junction_flanking_intron_seq(self):
        '''
        :param BS_LS_coordinates_path:
        :param flanking_bps:
        :return: two dictionaries
        1) junction_seq dictionary that stores the upper_intron, upper_exon, lower_exon, and lower_intron
        sequences with 50bps (default) each for a given LS or BS site with the key:
        chr|start|end|strand

        2) intronic_flanking_seq dictionary stores the flanking intronic sequence for the exon pairs with the key:
        chr|start|end|strand
        '''

        junction_seq = {}
        intronic_flanking_seq = {}

        ## retain the BS exons that have no valid boundaries
        BS_LS_coordinates_df = pd.read_csv(self.BS_LS_coordinates_path, sep='\t')
        for _, row in BS_LS_coordinates_df.iterrows():
            chrom, strand, start, end, label = row['chr'], row['strand'], int(row['start']), int(row['end']), \
                                               row['Splicing_type']
            # this key is unique for each instance in the BS_LS dataframe
            key = '|'.join([chrom, str(start), str(end), strand])

            # get the corresponding chromosome DNA sequence
            dna_seq = self.hg19_seq_dict[chrom]
            # extract the spliced genomic sequence assuming the positive strand
            spliced_seq_P = dna_seq[start: end].upper()
            # extract the upper_intron junction seq assuming the positive strand

            ### use 200 introns here by addition of 100bps, remove 100 to get the original data
            # upper_intron_P = dna_seq[start - self.flanking_bps-100: start].upper()
            upper_intron_P = dna_seq[start - self.flanking_junction_bps: start].upper()

            # extract the lower_intron junction seq assuming the positive strand
            # lower_intron_P = dna_seq[end: end + self.flanking_bps+100].upper()
            lower_intron_P = dna_seq[end: end + self.flanking_junction_bps].upper()

            # skip the rows that have 'Ns' in the upper_intron or lower_intron
            if 'N' in upper_intron_P or 'N' in lower_intron_P:
                print(f'{key} has N in the extracted junctions, belongs to {label}')
                continue

            # get the upper flanking sequence for the positive strand: instead of
            # using the up_end position use the start - 1000 as the starting point

            U_flanking_seq_P = dna_seq[start - self.flanking_intron_bps:start].upper()

            # get the lower flanking sequence using down_start + 1000 as the ending point
            L_flanking_seq_P = dna_seq[end:end + self.flanking_intron_bps].upper()

            # consider if the strand is -
            if strand == '-':
                ### get the junction sequence
                spliced_seq_N = self.reverse_complement(spliced_seq_P)
                upper_exon_N = spliced_seq_N[:self.flanking_junction_bps]
                lower_exon_N = spliced_seq_N[-self.flanking_junction_bps:]

                upper_intron_N = self.reverse_complement(lower_intron_P)
                lower_intron_N = self.reverse_complement(upper_intron_P)

                ### get the flanking intronic sequence
                U_flanking_seq = self.reverse_complement(L_flanking_seq_P)
                L_flanking_seq = self.reverse_complement(U_flanking_seq_P)

                intronic_flanking_seq[key] = {'U_flanking_seq': U_flanking_seq,
                                              'L_flanking_seq': L_flanking_seq,
                                              'label': label}

                junction_seq[key] = {'spliced_seq': spliced_seq_N,
                                     'upper_intron_' + str(self.flanking_junction_bps): upper_intron_N,
                                     'upper_exon_' + str(self.flanking_junction_bps): upper_exon_N,
                                     'lower_exon_' + str(self.flanking_junction_bps): lower_exon_N,
                                     'lower_intron_' + str(self.flanking_junction_bps): lower_intron_N,
                                     'label': label}

            else:
                upper_exon_P = spliced_seq_P[:self.flanking_junction_bps]

                lower_exon_P = spliced_seq_P[-self.flanking_junction_bps:]

                junction_seq[key] = {'spliced_seq': spliced_seq_P,
                                     'upper_intron_' + str(self.flanking_junction_bps): upper_intron_P,
                                     'upper_exon_' + str(self.flanking_junction_bps): upper_exon_P,
                                     'lower_exon_' + str(self.flanking_junction_bps): lower_exon_P,
                                     'lower_intron_' + str(self.flanking_junction_bps): lower_intron_P,
                                     'label': label}

                intronic_flanking_seq[key] = {'U_flanking_seq': U_flanking_seq_P,
                                              'L_flanking_seq': L_flanking_seq_P,
                                              'label': label}

        ### remove the repeative sequence from the junction_seq dictionary and then
        ### remove the overlapping sequence between BS and LS
        BS_junction_seq_dict = {}
        LS_junction_seq_dict = {}

        for i, j in junction_seq.items():
            if j['label'] == 'BS':
                BS_junction_seq_dict[i] = j['upper_intron_' + str(self.flanking_junction_bps)] + \
                                          j['upper_exon_' + str(self.flanking_junction_bps)] + \
                                          j['lower_exon_' + str(self.flanking_junction_bps)] + \
                                          j['lower_intron_' + str(self.flanking_junction_bps)]
            else:
                LS_junction_seq_dict[i] = j['upper_intron_' + str(self.flanking_junction_bps)] + \
                                          j['upper_exon_' + str(self.flanking_junction_bps)] + \
                                          j['lower_exon_' + str(self.flanking_junction_bps)] + \
                                          j['lower_intron_' + str(self.flanking_junction_bps)]

        # first get the 43 overlapping junction sequences and then filter the two dict from these sequences
        overlap_junction_seqs = set(BS_junction_seq_dict.values()).intersection(LS_junction_seq_dict.values())
        print(f'There are {len(overlap_junction_seqs)} overlapped flanking sequence from BS and LS  ')

        # remove these 43 overlapping junction sequences from BS_junction_seq_dict
        BS_junction_seq_dict_wo_overlap = {key: value for key, value in BS_junction_seq_dict.items() if
                                           value not in overlap_junction_seqs}

        ### check the repeated sequence in BS_junction_seq_dict_wo_overlap
        BS_repeated_seq_num = len(BS_junction_seq_dict_wo_overlap.values()) - len(
            set(BS_junction_seq_dict_wo_overlap.values()))
        print(f'There are {BS_repeated_seq_num} repeated BS sequences')

        # remove the duplicated junction sequences from BS by using the value as the key and then reverse it
        BS_tem_dict = {value: key for key, value in BS_junction_seq_dict_wo_overlap.items()}
        BS_res_dict = {value: key for key, value in BS_tem_dict.items()}

        # remove these 43 overlapping junction sequences from LS_junction_seq_dict
        LS_junction_seq_dict_wo_overlap = {key: value for key, value in LS_junction_seq_dict.items() if
                                           value not in overlap_junction_seqs}

        ### check the repeated sequence in LS_junction_seq_dict_wo_overlap
        LS_repeated_seq_num = len(LS_junction_seq_dict_wo_overlap.values()) - len(
            set(LS_junction_seq_dict_wo_overlap.values()))
        print(f'There are {LS_repeated_seq_num} repeated LS sequences')

        # remove the duplicated junction sequences from LS by using the value as the key and then reverse it
        LS_tem_dict = {value: key for key, value in LS_junction_seq_dict_wo_overlap.items()}
        LS_res_dict = {value: key for key, value in LS_tem_dict.items()}

        # merge two dicts
        BS_LS_res_dict = {**BS_res_dict, **LS_res_dict}

        # filter the junction_seq with the new keys from BS_LS_res_dict
        junction_seq_final = {key: junction_seq[key] for key in BS_LS_res_dict.keys()}

        # filter the intronic_flanking_seq with the new keys from BS_LS_res_dict
        intronic_flanking_seq_final = {key: intronic_flanking_seq[key] for key in BS_LS_res_dict.keys()}

        ## save the junction_seq and intronic_flanking_seq to json on the harddrive

        with open(f'{self.flanking_dict_folder}{self.junction_seq_json_name}.json', 'w') as f:
            json.dump(junction_seq_final, f)

        with open(f'{self.flanking_dict_folder}{self.flanking_seq_json_name}.json', 'w') as f:
            json.dump(intronic_flanking_seq_final, f)

    #         return junction_seq, intronic_flanking_seq

    def get_train_test_keys(self):

        if not os.path.exists(f'{self.flanking_dict_folder}{self.junction_seq_json_name}.json'):
            # invoke the function to write the BS_LS_junction_seq and BS_LS_intronic_flanking_seq to the drive
            self.get_junction_flanking_intron_seq()
            with open(f'{self.flanking_dict_folder}{self.junction_seq_json_name}.json') as f:
                BS_LS_junction_seq_dict = json.load(f)
        else:
            # read the BS_LS_junction_seq from harddrive instead of calling the function
            with open(f'{self.flanking_dict_folder}{self.junction_seq_json_name}.json') as f:
                BS_LS_junction_seq_dict = json.load(f)

        BS_exon_key_list = [key for key, value in BS_LS_junction_seq_dict.items() if value['label'] == 'BS']
        LS_exon_key_list = [key for key, value in BS_LS_junction_seq_dict.items() if value['label'] == 'LS']

        np.random.seed(42)
        # randomly select 8000 from both BS and LS key list and combine them as train

        # randomly select 11000 from both BS and LS key list and combine them
        BS_exon_train_keys = np.random.choice(BS_exon_key_list, size=11000, replace=False)

        ## different training size used for best base and rcm model selection (10000, 9000, 8000)
        BS_exon_train_keys_1 = BS_exon_train_keys[:self.training_size]
        ## 3000 used for combined model selection
        BS_exon_train_keys_2 = BS_exon_train_keys[self.training_size:]

        LS_exon_train_keys = np.random.choice(LS_exon_key_list, size=11000, replace=False)
        LS_exon_train_keys_1 = LS_exon_train_keys[:self.training_size]
        ## 3000 used for combined model selection
        LS_exon_train_keys_2 = LS_exon_train_keys[self.training_size:]

        BS_LS_exon_train_key1 = np.concatenate([BS_exon_train_keys_1, LS_exon_train_keys_1])
        BS_LS_exon_train_key2 = np.concatenate([BS_exon_train_keys_2, LS_exon_train_keys_2])

        # select the remaining keys from BS and LS key list as the test keys
        BS_exon_test_keys = set(BS_exon_key_list).difference(BS_exon_train_keys)
        LS_exon_test_keys = set(LS_exon_key_list).difference(LS_exon_train_keys)
        BS_LS_exon_test_keys = np.array(list(BS_exon_test_keys.union(LS_exon_test_keys)))

        return BS_LS_exon_train_key1, BS_LS_exon_train_key2, BS_LS_exon_test_keys


    def seq_to_matrix(self, input_seq):
        '''
            This function takes a DNA sequence and return a one-hot encoded matrix of 4 X N (length of input_seq)
        '''
        row_index = {'A': 0, 'G': 1, 'C': 2, 'T': 3}  # should exclude the 'Ns' in the input sequence

        # initialize the 4 X N 0 matrix:
        input_mat = np.zeros((4, len(input_seq)))

        for col_index, base in enumerate(input_seq):
            input_mat[row_index[base]][col_index] = 1
        return input_mat

    #### create all sequence features, rcm_features and a2i features silmaltineous here
    def seq_to_tensor(self, data_keys, rcm_folder, is_rcm=None, is_upper_lower_concat=None):
        '''
        :param data_keys:
        :param rcm_folder
        :param is_rcm: boolean value indicate if rcm features is genearated or not
        :param is_upper_lower_concat: boolean value indicates if upper and lower junction is concatenate or not
        :return: concatenated 2-d data for upper and lower seq for all the keys in the data_keys
        '''
        ### first get the BS_LS_junction_seq_dict;
        if not os.path.exists(f'{self.flanking_dict_folder}{self.junction_seq_json_name}.json'):
            self.get_junction_flanking_intron_seq()
            with open(f'{self.flanking_dict_folder}{self.junction_seq_json_name}.json') as f:
                BS_LS_junction_seq_dict = json.load(f)
        else:
            with open(f'{self.flanking_dict_folder}{self.junction_seq_json_name}.json') as f:
                BS_LS_junction_seq_dict = json.load(f)

        ### then get the BS_LS_flanking_seq_dict;
        if not os.path.exists(f'{self.flanking_dict_folder}{self.flanking_seq_json_name}.json'):
            self.get_junction_flanking_intron_seq()
            with open(f'{self.flanking_dict_folder}{self.flanking_seq_json_name}.json') as f:
                BS_LS_flanking_seq_dict = json.load(f)
        else:
            with open(f'{self.flanking_dict_folder}{self.flanking_seq_json_name}.json') as f:
                BS_LS_flanking_seq_dict = json.load(f)

        ### these two dictionary use the same keys
        ### list to store the concatenated upper and lower sequence one-hot encoding
        if is_upper_lower_concat:
            all_torch_feature_list = []

        ### list to store the upper and lower sequence one-hot encoding separately
        else:
            all_torch_upper_feature_list = []
            all_torch_lower_feature_list = []

        ### list to store rcm features if is_rcm true ####using for loops!!!!!!
        if is_rcm:
            ### tri-cnn for flanking, upper and lower separately
            flanking_rcm_scores = []
            upper_rcm_scores = []
            lower_rcm_scores = []

            flanking_rcm_dict_list = []
            upper_rcm_dict_list = []
            lower_rcm_dict_list = []

            seed_len_list = [5,7,9,11,13]

            #             for flanking_intron_len in [100, 200, 300, 400, 500,
            #                                         1000,1500,2000, 2500, 3000]: this is the next trial
            for flanking_intron_len in [100, 200, 300, 400, 500]:

                for seed_len in seed_len_list:
                    # print(flanking_intron_len, seed_len)
                    with open(os.path.join(rcm_folder,
                                           f'to_485_rcm_flanking_{flanking_intron_len}_bps_introns_{seed_len}mer.json')) as f:
                        flanking_rcm_dict = json.load(f)
                        flanking_rcm_dict_list.append(flanking_rcm_dict)
                    with open(os.path.join(rcm_folder,
                                           f'to_485_rcm_upper_{flanking_intron_len}_bps_introns_{seed_len}mer.json')) as f:
                        upper_rcm_dict = json.load(f)
                        upper_rcm_dict_list.append(upper_rcm_dict)
                    with open(os.path.join(rcm_folder,
                                           f'to_485_rcm_lower_{flanking_intron_len}_bps_introns_{seed_len}mer.json')) as f:
                        lower_rcm_dict = json.load(f)
                        lower_rcm_dict_list.append(lower_rcm_dict)

        ### list to store the corresponding labels: LS or BS
        all_label_list = []

        for key in data_keys:
            ### get the rcm features from the harddrive
            ### should produce the RCM features simultaneously with the seq feture for the same data point

            flanking_seqs = BS_LS_flanking_seq_dict[key]

            ## construct the rcm feature if is_rcm is True
            if is_rcm:
                flanking_rcm_kmer_list = [np.log(np.array(flanking_rcm[key][0]).reshape(5, 5) + 1) for flanking_rcm in
                                          flanking_rcm_dict_list]
                flanking_rcm_kmers = torch.from_numpy(np.concatenate(flanking_rcm_kmer_list, axis=1)).to(torch.float32)

                upper_rcm_kmer_list = [np.log(np.array(upper_rcm[key][0]).reshape(5, 5) + 1) for upper_rcm in
                                       upper_rcm_dict_list]
                upper_rcm_kmers = torch.from_numpy(np.concatenate(upper_rcm_kmer_list, axis=1)).to(torch.float32)

                lower_rcm_kmer_list = [np.log(np.array(lower_rcm[key][0]).reshape(5, 5) + 1) for lower_rcm in
                                       lower_rcm_dict_list]
                lower_rcm_kmers = torch.from_numpy(np.concatenate(lower_rcm_kmer_list, axis=1)).to(torch.float32)

                #                 rcm_values_concate = torch.from_numpy(np.stack([flanking_rcm_kmers,
                #                                                             upper_rcm_kmers,
                #                                                             lower_rcm_kmers], axis=0)).to(torch.float32)

                flanking_rcm_scores.append(flanking_rcm_kmers)
                upper_rcm_scores.append(upper_rcm_kmers)
                lower_rcm_scores.append(lower_rcm_kmers)

            ### working with the junction_seq starting here
            value = BS_LS_junction_seq_dict[key]
            # extract the poisitve (BS: as 1) and negative label (LS: as 0)

            ## make sure the label are the same for the same key and append the labe to the list
            label = value['label']
            assert label == flanking_seqs['label'], f"Same sequence key {key} with different labels"

            if label == 'BS':
                label = 1
            else:
                label = 0

            all_label_list.append(label)

            # concatenate the upper seq together and lower seq together for two separate CNN to process
            concatenated_upper_seq = value['upper_intron_{}'.format(str(self.flanking_junction_bps))] + \
                                     value['upper_exon_{}'.format(str(self.flanking_junction_bps))]
            concatenated_lower_seq = value['lower_exon_{}'.format(str(self.flanking_junction_bps))] + \
                                     value['lower_intron_{}'.format(str(self.flanking_junction_bps))]

            ### test whether want to concatenate the upper seq and lower seq together
            if is_upper_lower_concat:

                upper_lower_concat_seq = concatenated_upper_seq + concatenated_lower_seq
                upper_lower_concat_mat = self.seq_to_matrix(upper_lower_concat_seq)
                individual_upper_lower_torch = torch.from_numpy(upper_lower_concat_mat).to(torch.float32)
                all_torch_feature_list.append(individual_upper_lower_torch)

            else:

                individual_upper_mat = self.seq_to_matrix(concatenated_upper_seq)
                individual_lower_mat = self.seq_to_matrix(concatenated_lower_seq)

                # convert individual instance to torch
                individual_upper_torch = torch.from_numpy(individual_upper_mat).to(torch.float32)
                individual_lower_torch = torch.from_numpy(individual_lower_mat).to(torch.float32)

                all_torch_upper_feature_list.append(individual_upper_torch)
                all_torch_lower_feature_list.append(individual_lower_torch)

        all_torch_label = torch.tensor(all_label_list, dtype=torch.float32)  # .view(-1, 1)

        if is_rcm:
            all_torch_flanking_rcm_features = torch.stack(flanking_rcm_scores, dim=0)
            all_torch_upper_rcm_features = torch.stack(upper_rcm_scores, dim=0)
            all_torch_lower_rcm_features = torch.stack(lower_rcm_scores, dim=0)

        #             all_torch_rcm_feature = torch.stack(rcm_scores, dim=0)

        if is_upper_lower_concat:

            all_torch_feature = torch.stack(all_torch_feature_list, dim=0)

        else:
            all_torch_upper_feature = torch.stack(all_torch_upper_feature_list, dim=0)
            all_torch_lower_feature = torch.stack(all_torch_lower_feature_list, dim=0)

        ### return the tensor based on several requirement
        ## return only the upper, lower and rcm / upper lower, a2i / or both/ or just upper lower /or just concate

        if is_upper_lower_concat and not is_rcm:
            return all_torch_feature, all_torch_label

        if is_upper_lower_concat and is_rcm:
            return all_torch_feature, all_torch_flanking_rcm_features, all_torch_upper_rcm_features,\
                   all_torch_lower_rcm_features, all_torch_label

        if not is_upper_lower_concat and not is_rcm:
            return all_torch_upper_feature, all_torch_lower_feature, all_torch_label

        if not is_upper_lower_concat and is_rcm:
            return all_torch_upper_feature, all_torch_lower_feature, all_torch_flanking_rcm_features,\
                   all_torch_upper_rcm_features, all_torch_lower_rcm_features, all_torch_label


### Dataset preparation for Basemodel1
class BS_LS_upper_lower_concat_rcm(Dataset):
    def __init__(self, include_rcm, seq_upper_lower_feature, flanking_rcm, upper_rcm, lower_rcm, label):
        # construction of the map-style datasets
        # data loading
        self.include_rcm = include_rcm

        if self.include_rcm:
            self.x1 = seq_upper_lower_feature

            self.x2 = flanking_rcm
            self.x3 = upper_rcm
            self.x4 = lower_rcm

        else:
            self.x1 = seq_upper_lower_feature

        self.y = label

        self.n_samples = seq_upper_lower_feature.size()[0]

    def __getitem__(self, index):
        # dataset[0]
        if self.include_rcm:
            return self.x1[index], self.x2[index], self.x3[index], self.x4[index], self.y[index]
        else:
            return self.x1[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


### Dataset prepartion for Basemodel2
class BS_LS_upper_lower_rcm(Dataset):
    def __init__(self, include_rcm, seq_upper_feature, seq_lower_feature, flanking_rcm, upper_rcm, lower_rcm, label):
        # construction of the map-style datasets
        # data loading
        self.include_rcm = include_rcm

        if self.include_rcm:
            self.x1 = seq_upper_feature
            self.x2 = seq_lower_feature

            self.x3 = flanking_rcm
            self.x4 = upper_rcm
            self.x5 = lower_rcm

        else:
            self.x1 = seq_upper_feature
            self.x2 = seq_lower_feature

        self.y = label

        self.n_samples = seq_upper_feature.size()[0]

    def __getitem__(self, index):
        # dataset[0]
        if self.include_rcm:
            return self.x1[index], self.x2[index], self.x3[index], self.x4[index], self.x5[index], self.y[index]
        else:
            return self.x1[index], self.x2[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


### Dataset prepartion for RCM Scores
class RCM_Score(Dataset):
    def __init__(self, flanking_only=None, flanking_rcm=None, upper_rcm=None, lower_rcm=None, label=None):
        # construction of the map-style datasets
        # data loading
        self.flanking_only = flanking_only

        self.x1 = flanking_rcm
        self.x2 = upper_rcm
        self.x3 = lower_rcm

        self.y = label

        self.n_samples = flanking_rcm.size()[0]

    def __getitem__(self, index):
        # dataset[0]
        if self.flanking_only:
            return self.x1[index], self.y[index]
        else:
            return self.x1[index], self.x2[index], self.x3[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples
