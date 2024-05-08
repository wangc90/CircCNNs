import torch
import numpy as np
import json


class PPM_extraction():
    def __init__(self, train_instances, train_key, is_upper, is_lower, is_upper_lower_concat,
                 model_path, input_seq_len, padding_len, kernel_len,
                 ppm_file_name):
        '''

        :param train_instances: BS (positive instances) or LS (negative instances) apply the trained parameters to
        :param train_key: keys for the training data set
        :param is_upper: boolean value indicating whether the PPM is for upper
        :param is_lower: boolean value indicating whether the PPM is for lower
        :param is_upper_lower_concat: boolean value indicating whether the PPM is for upper_lower_concat
        :param model_path: best base model 1 or 2 depending on the upper, lower or upper_lower_concat
        :param input_seq_len: input sequence length for the model
        :param padding_len: padding length in the conv layer 1
        :param kernel_len: kernel length in the conv layer 2
        :param ppm_file_name: file name to save the extracted ppm based on the activation of conv layer 1
        '''
        self.train_instances = train_instances
        self.train_key = train_key
        self.is_upper = is_upper
        self.is_lower = is_lower
        self.is_upper_lower_concat = is_upper_lower_concat

        self.model_path = model_path
        self.input_seq_len = input_seq_len
        self.padding_len = padding_len
        self.kernel_len = kernel_len
        self.ppm_file_name = ppm_file_name

        ## the following four data will be automatically filled
        self.half_train_keys = None
        self.BS_upper_seqs = None
        self.BS_lower_seqs = None
        self.BS_upper_lower_concat_seqs = None
        self.activation = None
        self.subseq_starting_index_all_kernel = None
        self.sub_seq_all_kernels = None

        #### get the positive examples from the training dataset

    def get_BS_sequences(self, is_upper, is_lower, is_upper_lower_concat):

        with open('/home/wangc90/circRNA/circRNA_Data/BS_LS_data/flanking_dicts/BS_LS_junction_seq_100_bps.json') as f:
            BS_LS_junction_dict = json.load(f)

        BS_upper_sequences_list = []
        BS_lower_sequences_list = []

        BS_upper_lower_concat_sequences_list = []

        half_train_keys = []
        for key in self.train_key:

            label = BS_LS_junction_dict[key]['label']
            ## change this to 'LS' to get the PPM for negative instances
            if label == self.train_instances:
                half_train_keys.append(key)
                if is_upper:
                    BS_upper_sequences_list.append(BS_LS_junction_dict[key]['upper_intron_100'] + \
                                                   BS_LS_junction_dict[key]['upper_exon_100'])
                elif is_lower:
                    BS_lower_sequences_list.append(BS_LS_junction_dict[key]['lower_exon_100'] + \
                                                   BS_LS_junction_dict[key]['lower_intron_100'])

                else:
                    BS_upper_lower_concat_sequences_list.append(BS_LS_junction_dict[key]['upper_intron_100'] + \
                                                                BS_LS_junction_dict[key]['upper_exon_100'] + \
                                                                BS_LS_junction_dict[key]['lower_exon_100'] + \
                                                                BS_LS_junction_dict[key]['lower_intron_100'])
        self.half_train_keys = half_train_keys
        if is_upper:
            return BS_upper_sequences_list
        elif is_lower:
            return BS_lower_sequences_list
        else:
            return BS_upper_lower_concat_sequences_list


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

    def get_stacked_one_hot_seq_tensor(self, BS_seq_list):

        all_one_hot_seq_tensor_list = []
        for seq in BS_seq_list:
            one_hot_seq = self.seq_to_matrix(seq)
            one_hot_seq_tensor = torch.from_numpy(one_hot_seq).to(torch.float32)
            all_one_hot_seq_tensor_list.append(one_hot_seq_tensor)

        all_one_hot_seq_tensor_stacked = torch.stack(all_one_hot_seq_tensor_list, dim=0)

        return all_one_hot_seq_tensor_stacked

    def get_conv1_activation(self):

        best_base_model = torch.load(self.model_path)
        best_base_model.eval()

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        if self.is_upper or self.is_lower:
            ### register both upper and lower conv1 and evaluate the upper and lower
            ### seqs at the first conv layers based on the trained parameters in best base model2
            best_base_model.cnn_upper.conv1.register_forward_hook(get_activation('upper_conv1'))
            best_base_model.cnn_lower.conv1.register_forward_hook(get_activation('lower_conv1'))

            ### move the model to cpu
            best_base_model.to('cpu')

            self.BS_upper_seqs = self.get_BS_sequences(is_upper=True, is_lower=False, \
                                                       is_upper_lower_concat=False)
            self.BS_lower_seqs = self.get_BS_sequences(is_upper=False, is_lower=True, \
                                                       is_upper_lower_concat=False)

            X1 = self.get_stacked_one_hot_seq_tensor(self.BS_upper_seqs).to('cpu')
            X2 = self.get_stacked_one_hot_seq_tensor(self.BS_lower_seqs).to('cpu')

            output = best_base_model(X1, X2)

            self.activation = activation


        elif self.is_upper_lower_concat:
            best_base_model.cnn.conv1.register_forward_hook(get_activation('conv1'))

            ### move the model to cpu
            best_base_model.to('cpu')
            self.BS_upper_lower_concat_seqs = self.get_BS_sequences(is_upper=False, is_lower=False, \
                                                                    is_upper_lower_concat=True)

            X = self.get_stacked_one_hot_seq_tensor(self.BS_upper_lower_concat_seqs).to('cpu')

            output = best_base_model(X)

            self.activation = activation

    def get_subseq_starting_index(self):
        '''
            extract the subseq starting index that are corresponding to the maximal positive CNN layer 1
            activation values, should also consider the padding information,
            so the subsequences are not in the padded region
        '''

        self.get_conv1_activation()

        if self.is_upper:
            all_activation_values = self.activation['upper_conv1']
        elif self.is_lower:
            all_activation_values = self.activation['lower_conv1']

        elif self.is_upper_lower_concat:
            all_activation_values = self.activation['conv1']
        else:
            print('error')
            return

        subseq_starting_index_all_kernel = []

        ## we used the same padding strategy, so there are 200 activation values for each input sequence
        ## when input_seq is 200 bps
        ## however first padding_length and last padding_length of them are affected by the padded sequence
        #     negative_rows = 0
        for i in all_activation_values.permute(1, 0, 2):  ## from 11000 X 512 X 200 to 512 X 11000 X 200
            ### restrict the corresponding index to the non-padded sequence
            ### only focus on the subsequence that have max value > 0,
            ### values that are < 0 will be clipped by relu
            ### get the starting index that are derived from the real sequence
            max_index = torch.argmax(i[:, self.padding_len:self.input_seq_len - self.padding_len - self.kernel_len],
                                     dim=1).numpy()

            ## test if any of the value |in each row is positive in the junction seq;
            # this value must be positive
            positive_row = torch.any(i[:, self.padding_len:self.input_seq_len - self.padding_len - self.kernel_len] > 0,
                                     dim=1)

            ### convert the negative activation to np.nan and keep their position for easier sequence indexing later
            positive_row_bool = np.array([np.nan if is_positive == False else 1 for is_positive in positive_row])

            ## take their intersection ## convert the non positive max activation to negative num
            subseq_starting_index_individual_kernel = (max_index * positive_row_bool).tolist()
            subseq_starting_index_all_kernel.append(subseq_starting_index_individual_kernel)
        #     print(negative_rows)

        self.subseq_starting_index_all_kernel = subseq_starting_index_all_kernel

    def get_all_subseqs(self):

        self.get_subseq_starting_index()

        if self.is_upper:
            sequences = self.BS_upper_seqs

        elif self.is_lower:
            sequences = self.BS_lower_seqs

        elif self.is_upper_lower_concat:
            sequences = self.BS_upper_lower_concat_seqs

        else:
            print('error')
            return

        sub_seq_all_kernels = []
        ## loop through each kernel's max activation starting position index for all
        for subseq_indexs in self.subseq_starting_index_all_kernel:
            sub_seq_each_kernel = []
            for seq_index, starting_index in enumerate(subseq_indexs):
                if not np.isnan(starting_index):  # test for the row that have no positive activation values
                    sub_seq_each_kernel.append(
                        sequences[seq_index][int(starting_index):int(starting_index) + self.kernel_len])
            sub_seq_all_kernels.append(sub_seq_each_kernel)

        self.sub_seq_all_kernels = sub_seq_all_kernels

    def seq_to_matrix_for_tomtom(self, input_seq):
        '''
            This function takes a DNA sequence and return a one-hot encoded matrix of 4 X N (length of input_seq)
        '''
        row_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # should exclude the 'Ns' in the input sequence

        # initialize the 4 X N 0 matrix:
        input_mat = np.zeros((4, len(input_seq)))

        for col_index, base in enumerate(input_seq):
            input_mat[row_index[base]][col_index] = 1
        return input_mat

    def get_position_prob_matrix(self, subseq_list):
        '''
            build a poisition frequence matrix based on the subsequence extracted from each kernel's activation
            for each sequence get their one-hot representation and then add all the 4 X kernel matrix element wise
        '''
        ### {'A': 0, 'C': 1, 'G': 2, 'T': 3} as meme required alphabet order
        ### https://meme-suite.org/meme/doc/examples/sample-dna-motif.meme
        one_hot_seq_list = []
        for seq in subseq_list:
            one_hot_seq = self.seq_to_matrix_for_tomtom(seq)
            one_hot_seq_list.append(one_hot_seq)
        ## use z to accumulate the element wise sum
        z = np.zeros_like(one_hot_seq)
        for one_hot in one_hot_seq_list:
            z += one_hot

        #         if (11000 - np.sum(z, axis=0)[0]) != 0:
        #             print(11000 - np.sum(z, axis=0)[0])
        return z / np.sum(z, axis=0)

    ### write the position prob matrix to txt file

    def write_out_PPM(self):

        self.get_all_subseqs()

        with open(f'/home/wangc90/circRNA/circRNA_Data/model_outputs/Extracted_motifs/{self.ppm_file_name}.txt',
                  'w') as f:
            f.write('MEME version 4.9.0\n\n'
                    'ALPHABET= ACGT\n\n'
                    'strands: + -\n\n'
                    'Background letter frequencies (from uniform background):\n'
                    'A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n')

            for kernel_number in range(len(self.sub_seq_all_kernels)):
                ### loop through the position_prob_matrix for each kernel
                f.write(f'MOTIF {kernel_number}\n')
                f.write(
                    f"letter-probability matrix: alength= 4 w= {self.kernel_len} nsites= {self.kernel_len} E= 1e-6\n")
                for line in self.get_position_prob_matrix(self.sub_seq_all_kernels[kernel_number]).T:
                    f.write('\t'.join([str(item) for item in line]) + '\n')
                f.write('\n')
