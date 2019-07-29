import json
import torch

def right_align(sequences, lengths):
    
    aligned = torch.zeros(sequences.size())
    n = sequences.size()[0]
    m = sequences.size()[1]

    for i in range(n):
        if lengths[i] > 0:
            aligned[i][m-lengths[i]:] = sequences[i][:lengths[i]]
    
    return aligned

def getopt(opt=None, key=None, default_value=None):
    try:
        if default_value == None and (opt == None or opt[key] == None):
            raise Exception('required key', key, 'is not provided')
        if opt == None:
            return default_value
        v = opt[key]
        if v == None:
            v = default_value
        return v

    except Exception as err:
        print(err)
        raise

def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data

def write_json(path, j):
    with open(path, 'w') as json_file:
        json.dump(j, json_file)
    
def dict_average(dicts):
    dictionary = {}
    n = 0
    for d in dicts:
        for k, v in d:
            if dictionary[k] == None :
                dictionary[k] = 0
            dictionary[k] += v
        n += 1
    
    for k, v in dictionary:
        dictionary[k] /= n

    return dictionary

def count_keys(t):
    return len(t)

def average_values(t):
    n = 0
    vsum = 0
    for k, v in t:
        vsum += v
        n += 1
    return vsum / n