import h5py
import json
import utils
import numpy as np
import torch

class Dataloader(object):
    def __init__(self, input_json_file_path, input_ques_h5_path):
        
        print('Reading', input_json_file_path)
        
        with open(input_json_file_path) as input_file:
            data_dict = json.load(input_file)
        
        for k, v in data_dict:
            self.__dict__[k] = v
        
        self.vocab_size = 0
        for i, w in self.ix_to_word:
            self.vocab_size += 1

        print('DataLoader loading h5 question file:', input_ques_h5_path)
        qa_data = h5py.File(input_ques_h5_path, 'r')
        self.ques_train = qa_data['ques_train']
        self.ques_len_train = qa_data['ques_length_train']
        self.ques_id_train = qa_data['ques_cap_id_train']

        self.label_train = qa_data['ques1_train']
        self.label_len_train = qa_data['ques1_length_train']

        self.train_id = 0
        self.seq_length = self.ques_train.shape[1]

        print('self.ques_train.shape[0]', self.ques_train.shape[0])

        self.ques_test = qa_data['ques_test']
        self.ques_len_test = qa_data['ques_length_test']
        self.ques_id_test = qa_data['ques_cap_id_test']

        self.label_test = qa_data['ques1_test']
        self.label_len_test = qa_data['ques1_length_test']

        self.test_id = 0

        print('self.ques_test.shape[0]', self.ques_test.shape[0])
        qa_data.close()

    def next_batch(self, batch_size, gpuid=-1):
        start_id = self.train_id
        if start_id + batch_size <= self.ques_train.shape[0]:
            end_id = start_id + batch_size
        else:
            print('end of epoch')
            self.train_id = 0
            start_id = self.train_id
            end_id = start_id + batch_size
        
        ques = torch.from_numpy(self.ques_train[start_id:end_id])
        label = torch.from_numpy(self.label_train[start_id:end_id])
        ques_id = torch.from_numpy(self.ques_id_train[start_id:end_id])
        
        if gpuid >= 0 :
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            ques = ques.to(device)
            label = label.to(device)
            ques_id = ques_id.to(device)

        self.train_id += end_id - start_id
        return (ques, label, ques_id)

    def next_batch_eval(self, batch_size, gpuid=-1):
        start_id = self.test_id
        end_id = min(start_id + batch_size, self.ques_test.shape[0])

        ques = torch.from_numpy(self.ques_test[start_id:end_id])
        label = torch.from_numpy(self.label_test[start_id:end_id])
        ques_id = torch.from_numpy(self.ques_id_test[start_id:end_id])
        
        if gpuid >= 0 :
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            ques = ques.to(device)
            label = label.to(device)
            ques_id = ques_id.to(device)

        self.test_id += end_id - start_id
        return (ques, label, ques_id)

    def getVocab(self):
        return self.ix_to_word

    def getVocabSize(self):
        return self.vocab_size

    def resetIterator(self, split):
        if split == 1 :
            self.train_id = 0
        if split == 2 :
            self.test_id = 0
        
    def getDataNum(self, split):
        if split == 1:
            return self.ques_train.shape[0]

        if split == 2:
            return self.ques_test.shape[0]

    def getSeqLength(self):
        return self.seq_length
