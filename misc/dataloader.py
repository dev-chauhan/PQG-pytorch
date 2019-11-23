import h5py
import json
import misc.utils as utils
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

class Dataloader(data.Dataset):

    def __init__(self, input_json_file_path, input_ques_h5_path):
        
        super(Dataloader, self).__init__()
        print('Reading', input_json_file_path)
        
        with open(input_json_file_path) as input_file:
            data_dict = json.load(input_file)
        
        self.ix_to_word = {}
        
        for k in data_dict['ix_to_word']:
            self.ix_to_word[int(k)] = data_dict['ix_to_word'][k]
        
        self.UNK_token = 0
        
        if 0 not in self.ix_to_word:
            self.ix_to_word[0] = '<UNK>'
        
        else : 
            raise Exception
        
        self.EOS_token = len(self.ix_to_word)
        self.ix_to_word[self.EOS_token] = '<EOS>'
        self.PAD_token = len(self.ix_to_word)
        self.ix_to_word[self.PAD_token] = '<PAD>'
        self.SOS_token = len(self.ix_to_word)
        self.ix_to_word[self.SOS_token] = '<SOS>'
        self.vocab_size = len(self.ix_to_word)
        print('DataLoader loading h5 question file:', input_ques_h5_path)
        qa_data = h5py.File(input_ques_h5_path, 'r')
        
        ques_id_train = torch.from_numpy(qa_data['ques_cap_id_train'][...].astype(int))
        
        ques_train, ques_len_train = self.process_data(torch.from_numpy(qa_data['ques_train'][...].astype(int)), torch.from_numpy(qa_data['ques_length_train'][...].astype(int)))
        
        label_train, label_len_train = self.process_data(torch.from_numpy(qa_data['ques1_train'][...].astype(int)), torch.from_numpy(qa_data['ques1_length_train'][...].astype(int)))

        self.train_id = 0
        self.seq_length = ques_train.size()[1]

        print('self.ques_train.shape[0]', ques_train.size()[0])


        ques_test, ques_len_test = self.process_data(torch.from_numpy(qa_data['ques_test'][...].astype(int)), torch.from_numpy(qa_data['ques_length_test'][...].astype(int)))
        
        label_test, label_len_test = self.process_data(torch.from_numpy(qa_data['ques1_test'][...].astype(int)), torch.from_numpy(qa_data['ques1_length_test'][...].astype(int)))

        ques_id_test = torch.from_numpy(qa_data['ques_cap_id_test'][...].astype(int))

        self.test_id = 0

        print('self.ques_test.shape[0]', ques_test.size()[0])
        qa_data.close()

        self.ques = torch.cat([ques_train, ques_test])
        self.len = torch.cat([ques_len_train, ques_len_test])
        self.label = torch.cat([label_train, label_test])
        self.label_len = torch.cat([label_len_train, label_len_test])
        self.id = torch.cat([ques_id_train, ques_id_test])

    def process_data(self, data, data_len):
        N = data.size()[0]
        new_data = torch.zeros(N, data.size()[1] + 2, dtype=torch.long) + self.PAD_token
        for i in range(N):
            new_data[i, 1:data_len[i]+1] = data[i, :data_len[i]]
            new_data[i, 0] = self.SOS_token
            new_data[i, data_len[i]+1] = self.EOS_token
            data_len[i] += 2
        return new_data, data_len
    
    def __len__(self):
        return self.len.size()[0]

    def __getitem__(self, idx):
        return (self.ques[idx], self.len[idx], self.label[idx], self.label_len[idx], self.id[idx])

    # def getVocab(self):
    #     self.ix_to_word['0'] = ''
    #     return self.ix_to_word

    def getVocabSize(self):
        return self.vocab_size

    # def resetIterator(self, split):
    #     if split == 1 :
    #         self.train_id = 0
    #     if split == 2 :
    #         self.test_id = 0
        
    def getDataNum(self, split):
        if split == 1:
            return 100000

        if split == 2:
            return 30000

    def getSeqLength(self):
        return self.seq_length
