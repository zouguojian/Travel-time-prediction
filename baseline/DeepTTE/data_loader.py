import time
import utils

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ujson as json

"""
Each JSON file contains a certain number of trips represented as a dict with the following keys:
Features associated with entire trip i.e. len(dict[key]) == 1 (categorical unless o/w stated)
1. driverID
2. weekID
3. dateID
5. timeID (start time of trip)
5. dist - continuous
6. time - continuous (ground truth, total travel time) 

Features associated with each ping i.e. len(dict[key]) >= 1(all continuous)
1. lats
2. lngs
3. dist_gap
4. time_gap
5. states (optional)
"""


class MySet(Dataset):
    def __init__(self, input_file):
        # self.content = open('./data/' + input_file, 'r').readlines
        # self.content = map(lambda x: json.loads(x), self.content)
        # self.lengths = map(lambda x: len(x['lngs']), self.content)

        ### Python 3 conversion
        with open('./data/' + input_file, 'r') as f:
            ### readLines() outputs a list of strings, each strings represents a dict with \n at the end
            self.content = f.readlines()
            ### json.loads() returns each string as a dict, content is now a map object i.e. list of dicts
            self.content = list(map(lambda x: json.loads(x), self.content))
            ### gets the number of trajectories in each trip, lengths is a map object i.e. list of int
            self.lengths = list(map(lambda x: len(x['lngs']), self.content))

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)

### This function is called when setting up the PyTorch dataloader
### Not quite sure how this works together with BatchSampler, but guess it is called on each batch
def collate_fn(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']
    traj_attrs = ['lngs', 'lats', 'states', 'time_gap', 'dist_gap']

    attr, traj = {}, {}

    ### item refers to each trip, len(item['lngs']) would return length of each trip
    ### lens is an array of length of each trip in the batch
    lens = np.asarray([len(item['lngs']) for item in data])

    ### Since these features are continuous, then normalise them
    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)

    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])

    for key in traj_attrs:
        # pad to the max length
        ### Each element in seqs is a list of values for that variable
        seqs = np.asarray([item[key] for item in data])
        ### Creates a mask according to length of each trip wrt maximum trip length
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype = np.float32)
        ### padded is a 2D array containing the padded sequence of values for each trip
        ### Each row represents a trip padded to the maximum length of trips in the batch
        ### Alternatively could use torch.nn.utils.rnn.pad_sequence here
        padded[mask] = np.concatenate(seqs)

        if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
            padded = utils.normalize(padded, key)

        ### Convert to torch tensor
        padded = torch.from_numpy(padded).float()
        traj[key] = padded

    lens = lens.tolist()
    traj['lens'] = lens

    return attr, traj


### This function is evoked when setting up the PyTorch dataloader
class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = list(range(self.count))

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        ### Sorting is done here to prepare inputs for the temporal LSTM layer further down
        ### rnn.packed_padded_sequence with enforce_sorted=True is used in SpatioTemporal.py
        ### This requires the inputs to be sorted according to descending length, which is done here
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


def get_loader(input_file, batch_size):
    dataset = MySet(input_file = input_file)
    ### dataset is a self-defined MySet object of training data with attributes content and lengths (of type map)

    batch_sampler = BatchSampler(dataset, batch_size)

    data_loader = DataLoader(dataset = dataset, \
                             batch_size = 1, \
                             collate_fn = lambda x: collate_fn(x), \
                             num_workers = 4,
                             batch_sampler = batch_sampler,
                             pin_memory = True)
    return data_loader