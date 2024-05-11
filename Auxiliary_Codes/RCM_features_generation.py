import pandas as pd
import sys
import numpy as np
import os
from collections import defaultdict
import json
from itertools import product
import ray
from itertools import combinations
os.environ['RAY_worker_register_timeout_seconds'] = '60'


@ray.remote
class RCS_finder():
    '''
        This is a python class that can find the imperfect reverse complementary sequences (RCS) within
        a intronic sequence ranging from 50bps to 50kbps; it can also find the imperfect RCS of two
        flanking introns each ranging from 50bps to 50kbps.

        input_seq1: the Up flanking intron;
        input_seq2: the Lower flanking intron;
        is_flanking_introns: boolean value indicating whether dealing with two flanking introns
        if is_flanking_introns == False, the input_seq2 will be ignored and only input_seq1 will be used;
        seed_len: seed length for the initial RCS search
        seq_fraction_of_spacer: the minimal portion of the seq length between the two paired seed region
                                when dealing with one intron (between 0 to 1); set spacer to 0 if
                                spacer length is not used for filtering;
        allowed_seed_mismatch: allowed the number of mismatch in the seed region: default is 1 (i.e., 2 score)
    '''

    def __init__(self, key=None, input_seq1=None, input_seq2=None, is_flanking_introns=None,
                 is_upper_intron=None, seq_fraction_of_spacer=None,
                 seed_len=None, allowed_seed_mismatch=None):

        ### input_seq1 is the upper intronic sequence
        ### input_seq2 is the lower intronic sequence
        self.key = key
        self.input_seq1 = input_seq1
        self.input_seq2 = input_seq2
        self.is_flanking_introns = is_flanking_introns
        self.is_upper_intron = is_upper_intron
        self.seq_fraction_of_spacer = seq_fraction_of_spacer
        self.seed_len = seed_len
        self.allowed_seed_mismatch = allowed_seed_mismatch

    #         self.num_rcm_kmers = None
    #         self.num_rcm_kmers_per_10000_comp = None

    def base_conversion(self, seq):
        '''
            convert the input sequence into the numerical representation
        '''
        base_mapping = {'A': 1, 'T': -1, 'C': 1j, 'G': -1j, 'N': 0}
        if len(seq) >= 2:
            seq_map = np.array([base_mapping[i] for i in seq])
        else:
            seq_map = base_mapping[seq]
        return seq_map

    def get_sub_seq_score(self, seq):

        '''
            This function starts with a input sequence and return
            the cummulative sum of the subsequence of length min_seq
        '''
        # get the augumented input sequence
        aug_seq = 'N' + seq
        # convert the augumented sequence into the numerical representation
        aug_seq_M = self.base_conversion(aug_seq)
        # get the cumulative sum of the augumented sequence
        aug_seq_cum_M = aug_seq_M.cumsum()
        # get the cumulative sum of the original input sequence
        seq_cum_M = aug_seq_cum_M[1:]
        # get subsequence score with length=self.seed_len
        sub_seq_score = seq_cum_M[self.seed_len - 1:] - aug_seq_cum_M[:len(aug_seq) - self.seed_len]
        return sub_seq_score

    def get_allowed_window_index(self):
        '''
            This function takes in the subseq_scores, and a fraction number (i.e., betwen 0 and 1) and return
            the allowed pair of window index that can be used for further subseqs verification
        '''

        if self.is_flanking_introns:
            #             print(self.input_seq1)
            #             print(self.input_seq2)
            # deal with flanking introns
            subseq_score1 = self.get_sub_seq_score(self.input_seq1).astype(np.complex64)
            subseq_score2 = self.get_sub_seq_score(self.input_seq2).astype(np.complex64)

            # put all possible pair-wise subseq addition score in the matrix
            # first subseq as the column, second subseq as the row
            subseq_scores_sum = subseq_score1.reshape((-1, 1)) + subseq_score2.reshape((1, -1))

            # get the sum of the absolute value of imaginary part and real part
            subseq_scores_abs_sum_mat = abs(subseq_scores_sum.imag) + abs(subseq_scores_sum.real)

            # get the index for the pair of subseqs that potentially have at most 1 mismatch (i.e., abs sum <= 2)
            first_window, second_window = np.where(subseq_scores_abs_sum_mat <= self.allowed_seed_mismatch)

            return first_window, second_window

        elif self.is_upper_intron:
            seq = self.input_seq1
        else:
            seq = self.input_seq2

        #         print(seq)

        # deal with one intron only, use the seq
        subseq_scores = self.get_sub_seq_score(seq)

        ### this step is fast
        subseq_scores = subseq_scores.astype(np.complex64)

        #### this step takes about 7 seconds
        # put all possible pair-wise subseq addition score in the matrix
        subseq_scores_sum = subseq_scores.reshape((-1, 1)) + subseq_scores.reshape((1, -1))

        #### this step takes about 11 seconds
        # get the sum of the absolute value of imaginary part and real part
        subseq_scores_abs_sum_mat = abs(subseq_scores_sum.imag) + abs(subseq_scores_sum.real)

        #### this step takes about 9 seconds
        # obtain the lower triangular index including the diagnoal
        tril_index = np.tril_indices(subseq_scores_abs_sum_mat.shape[0])

        #### this step takes about 7 seconds
        # change the lower triangular value to be bigger than 2, thus don't need to filter due to the  matrix symetry
        subseq_scores_abs_sum_mat[tril_index] = self.allowed_seed_mismatch + 1

        #### this step takes about 9 seconds
        # get the index for the pair of subseqs that potentially have at most 1 mismatch (i.e., abs sum <= 2)
        first_window, second_window = np.where(subseq_scores_abs_sum_mat <= self.allowed_seed_mismatch)

        ### this step takes about 0.9 seconds
        # get the window distance for each pair of subseqs
        window_num_diff = second_window - first_window

        #### this step takes about 0.7 seconds
        # get the base pair distance for each pair of subseqs # 10 is the default arm length
        spacer = window_num_diff - self.seed_len

        ### this step takes about 0.3 seconds; set fraction_of_spacer at least to 0 to
        ### avoid the overlapping windows
        # base pair distance for each pair of subseqs at least half the total seq length
        allowed_index = (spacer >= len(seq) * self.seq_fraction_of_spacer)

        ### these two steps take about 0.6 seconds
        # get the allowed window index
        allowed_first_window = first_window[allowed_index]
        allowed_second_window = second_window[allowed_index]

        return allowed_first_window, allowed_second_window

    def subseq_validity_check(self):
        '''
            This function is used to check wether the subseq pairs identified in get_allowed_window_index function
            is valid or not by calling the check_valid_subseq_pairs function,
            This function also return the relative location of the valid rcm kmers as well as number of
            kmers per 10000 comparison
        '''
        first_window, second_window = self.get_allowed_window_index()

        valid_subseq_pairs_list = []

        complement_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'N': 'N'}

        ### check to see if dealing with one introns or flanking introns
        if self.is_flanking_introns:
            input_seq1 = self.input_seq1
            input_seq2 = self.input_seq2

        elif self.is_upper_intron:
            input_seq1 = self.input_seq1
            input_seq2 = self.input_seq1
        else:
            input_seq1 = self.input_seq2
            input_seq2 = self.input_seq2

        for first_index, second_index in zip(first_window, second_window):

            # set it for each window pair
            seed_mismatch_score = 0

            first_subseq, second_subseq = input_seq1[first_index:self.seed_len + first_index], \
                                          input_seq2[second_index:self.seed_len + second_index]

            # check for the two ends first to make sure end is matching
            if first_subseq[0] == complement_dict[second_subseq[-1]]:
                # check the interior subseq
                for i in range(1, self.seed_len):
                    if first_subseq[i] != complement_dict[second_subseq[-i - 1]]:
                        seed_mismatch_score += 2
                        # if total mismatch already bigger than 2 abort the further checking
                        if seed_mismatch_score > self.allowed_seed_mismatch:
                            break

                if seed_mismatch_score <= self.allowed_seed_mismatch:
                    valid_subseq_pairs_list.append((first_subseq, first_index, second_subseq, \
                                                    second_index, seed_mismatch_score))

        #         print(valid_subseq_pairs_list)
        ### The following code: extract the rcm kmer location distribution relative
        ### to the corresponding upper or lower intron

        ### create interval list for upper and lower introns based on the upper and lower intron length

        upper_equal_space_list = np.linspace(start=0, stop=len(input_seq1), num=5 + 1)

        lower_equal_space_list = np.linspace(start=0, stop=len(input_seq2), num=5 + 1)

        upper_interval_list = []
        lower_interval_list = []

        for i in range(len(upper_equal_space_list) - 1):
            upper_interval = (upper_equal_space_list[i], upper_equal_space_list[i + 1])
            upper_interval_list.append(upper_interval)

        for i in range(len(lower_equal_space_list) - 1):
            lower_interval = (lower_equal_space_list[i], lower_equal_space_list[i + 1])
            lower_interval_list.append(lower_interval)

        ### get the combination of interval from upper and lower introns 5 x 5 interval in this case
        ### if num = 6
        interval_combinations = list(product(upper_interval_list, lower_interval_list))

        ## save this interval_combinations to self.interval_combinations for the distribution of extended subseq pairs
        self.interval_combinations = interval_combinations

        ### get the upper and lower rcm kmer position pairs
        upper_lower_rcm_kmer_pos_pairs = []
        for i in valid_subseq_pairs_list:
            upper_lower_rcm_kmer_pos_pairs.append((i[1], i[3]))

        ### calculate the total number of rcm kmer in each combination of intervals
        ### and store them in num_rcm_kmer_cross_intervals list
        num_rcm_kmer_cross_intervals = []

        for interval_comb in self.interval_combinations:
            upper_interval_pos = interval_comb[0]
            lower_interval_pos = interval_comb[1]
            ## check if the upper in upper_interval_pos and lower in lower_interval_pos
            ## upper or lower + 0.5 * self.seed length was used as the center of rcm kmer
            num_rcm_kmer_cross_intervals.append(sum([(upper + 0.5 * self.seed_len >= upper_interval_pos[0] and \
                                                      upper + 0.5 * self.seed_len < upper_interval_pos[1]) and \
                                                     (lower + 0.5 * self.seed_len >= lower_interval_pos[0] and \
                                                      lower + 0.5 * self.seed_len < lower_interval_pos[1]) \
                                                     for upper, lower in upper_lower_rcm_kmer_pos_pairs]))

        joint_rcm_kmer_dist = np.array(num_rcm_kmer_cross_intervals)

        return self.key, list(joint_rcm_kmer_dist)


def rcs_flanking_introns_ray_chunk(seq_list, seed_len):
    rcs_list = []
    for key, value in seq_list:
        seq1 = value['U_flanking_seq']
        #         print(len(seq1))
        seq2 = value['L_flanking_seq']
        #         print(len(seq2))
        rcs_list.append(RCS_finder.remote(key=key, input_seq1=seq1, input_seq2=seq2,
                                          is_flanking_introns=True,
                                          seed_len=seed_len, is_upper_intron=False,
                                          seq_fraction_of_spacer=0,
                                          allowed_seed_mismatch=0))

    results = ray.get([rcs.subseq_validity_check.remote() for rcs in rcs_list])
    return results


def loop_flanking_introns_ray_chunk(BS_LS_flanking_seq_dict, chunk_keys, rcm_dump_folder, seed_len, flanking_len):
    '''
        This function loops chunk_keys and call the rcs_flanking_introns_ray_chunk function
        in each loop and collect the results in a dict and dump them in rcm_dump_folder
    '''
    rcs_flanking_introns_results = []
    result_dict = defaultdict(list)

    for i in range(len(chunk_keys)):
        seq_key = chunk_keys[i]

        seq_list = [(key, BS_LS_flanking_seq_dict[key]) for key in seq_key]

        result = rcs_flanking_introns_ray_chunk(seq_list=seq_list, seed_len=seed_len)

        rcs_flanking_introns_results.append(result)

    #         print(rcs_flanking_introns_results)

    flat_list = [item for chunk in rcs_flanking_introns_results for item in chunk]

    for item in flat_list:
        rcm_num = [float(i) for i in item[1]]
        result_dict[item[0]].append(rcm_num)

    #         rcm_num_norm = [float(i) for i in item[2]]
    #         result_dict[item[0]].append(rcm_num_norm)

    file_name = f"to_{i + 1}_rcm_flanking_{flanking_len}_bps_introns_{seed_len}mer.json"

    with open(os.path.join(rcm_dump_folder, file_name), 'w') as f:
        json.dump(result_dict, f)

    print(f'flanking_rcm {i + 1} finished')

    ray.shutdown()


def rcs_upper_introns_ray_chunk(seq_list, seed_len):
    rcs_list = []
    for key, value in seq_list:
        seq1 = value['U_flanking_seq']
        seq2 = value['L_flanking_seq']
        rcs_list.append(RCS_finder.remote(key=key, input_seq1=seq1, input_seq2=seq2,
                                          is_flanking_introns=False,
                                          seed_len=seed_len, is_upper_intron=True,
                                          seq_fraction_of_spacer=0,
                                          allowed_seed_mismatch=0))

    results = ray.get([rcs.subseq_validity_check.remote() for rcs in rcs_list])
    return results


def loop_upper_introns_ray_chunk(BS_LS_flanking_seq_dict, chunk_keys, rcm_dump_folder, seed_len, flanking_len):
    '''
        This function loops chunk_keys and call the rcs_upper_introns_ray_chunk function
        in each loop and collect the results in a list
    '''
    rcs_upper_introns_results = []
    result_dict = defaultdict(list)

    for i in range(len(chunk_keys)):
        seq_key = chunk_keys[i]

        seq_list = [(key, BS_LS_flanking_seq_dict[key]) for key in seq_key]

        result = rcs_upper_introns_ray_chunk(seq_list=seq_list, seed_len=seed_len)

        rcs_upper_introns_results.append(result)

    #         print(rcs_upper_introns_results)

    flat_list = [item for chunk in rcs_upper_introns_results for item in chunk]

    for item in flat_list:
        rcm_num = [float(i) for i in item[1]]
        result_dict[item[0]].append(rcm_num)

    file_name = f"to_{i + 1}_rcm_upper_{flanking_len}_bps_introns_{seed_len}mer.json"

    with open(os.path.join(rcm_dump_folder, file_name), 'w') as f:
        json.dump(result_dict, f)

    print(f'upper_rcm {i + 1} finished')

    ray.shutdown()


def rcs_lower_introns_ray_chunk(seq_list, seed_len):
    rcs_list = []
    for key, value in seq_list:
        seq1 = value['U_flanking_seq']
        seq2 = value['L_flanking_seq']
        rcs_list.append(RCS_finder.remote(key=key, input_seq1=seq1, input_seq2=seq2,
                                          is_flanking_introns=False,
                                          seed_len=seed_len, is_upper_intron=False,
                                          seq_fraction_of_spacer=0,
                                          allowed_seed_mismatch=0))

    results = ray.get([rcs.subseq_validity_check.remote() for rcs in rcs_list])
    return results


def loop_lower_introns_ray_chunk(BS_LS_flanking_seq_dict, chunk_keys, rcm_dump_folder, seed_len, flanking_len):
    '''
        This function loops chunk_keys and call the rcs_lower_introns_ray_chunk function
        in each loop and collect the results in a list
    '''
    rcs_lower_introns_results = []
    result_dict = defaultdict(list)

    for i in range(len(chunk_keys)):
        seq_key = chunk_keys[i]

        seq_list = [(key, BS_LS_flanking_seq_dict[key]) for key in seq_key]

        result = rcs_lower_introns_ray_chunk(seq_list=seq_list, seed_len=seed_len)

        rcs_lower_introns_results.append(result)
    #         print(rcs_lower_introns_results)

    flat_list = [item for chunk in rcs_lower_introns_results for item in chunk]

    for item in flat_list:
        rcm_num = [float(i) for i in item[1]]
        result_dict[item[0]].append(rcm_num)

    #         rcm_num_norm = [float(i) for i in item[2]]
    #         result_dict[item[0]].append(rcm_num_norm)

    file_name = f"to_{i + 1}_rcm_lower_{flanking_len}_bps_introns_{seed_len}mer.json"
    with open(os.path.join(rcm_dump_folder, file_name), 'w') as f:
        json.dump(result_dict, f)

    print(f'lower_rcm {i + 1} finished')

    ray.shutdown()


### Point to the folder where you want to save the rcm_scores
rcm_folder = '/home/wangc90/circRNA/circRNA_Data/BS_LS_data/flanking_dicts/rcm_scores/'

### Put the results for different flanking window size in the list
BS_LS_flanking_seq_dict_list = []

for flanking_intron_len in [200, 300]:
    ### the flanking sequence file
    with open(f'/home/wangc90/circRNA/circRNA_Data/BS_LS_data/flanking_dicts/BS_LS_intronic_flanking_seq_{flanking_intron_len}_bps.json') as f:
        BS_LS_flanking_seq_dict = json.load(f)
        BS_LS_flanking_seq_dict_list.append(BS_LS_flanking_seq_dict)

all_keys = [key for key in BS_LS_flanking_seq_dict_list[0].keys()]

### Execute a chunk (50) of exon pairs at the same time
chunk_keys_flanking_introns = []
for i in range(0, len(all_keys), 50):
    chunk_keys_flanking_introns.append(all_keys[i:i + 50])

chunk_keys_within_introns = []
for i in range(0, len(all_keys), 50):
    chunk_keys_within_introns.append(all_keys[i:i + 50])

### define the seed length
seed_len_list = [5, 7, 9, 11, 13]
flanking_len_list = [200, 300]

### loop through them
for index, value in enumerate(BS_LS_flanking_seq_dict_list):

    for seed_len in seed_len_list:
        print(f'get flanking rcm feature for {flanking_len_list[index]} bps intron with seed length {seed_len}')

        loop_flanking_introns_ray_chunk(BS_LS_flanking_seq_dict=value, chunk_keys=chunk_keys_flanking_introns,
                                        rcm_dump_folder=rcm_folder, seed_len=seed_len,
                                        flanking_len=flanking_len_list[index])
        # get rcm feature for upper introns:
        print(f'get upper rcm feature for {flanking_len_list[index]} bps intron with seed length {seed_len}')

        loop_upper_introns_ray_chunk(BS_LS_flanking_seq_dict=value, chunk_keys=chunk_keys_within_introns,
                                     rcm_dump_folder=rcm_folder, seed_len=seed_len,
                                     flanking_len=flanking_len_list[index])
        # #     ### get rcm feature for lower introns:
        print(f'get lower rcm feature for {flanking_len_list[index]} bps intron with seed length {seed_len}')

        loop_lower_introns_ray_chunk(BS_LS_flanking_seq_dict=value, chunk_keys=chunk_keys_within_introns,
                                     rcm_dump_folder=rcm_folder, seed_len=seed_len,
                                     flanking_len=flanking_len_list[index])