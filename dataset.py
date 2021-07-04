# -*- coding: utf-8 -*-
import torch 
from torch.utils.data import Dataset 
import numpy as np
import pickle
from utils.utils_wav import amplitude_modulate, truncate


class CustomDataset(Dataset):
    def __init__(self, mode, test_sess):
        self.mode = mode
        self.data = []

        if self.mode == 'train':
            for i in range(1, 6):
                if i != test_sess:
                    with open('Session' + str(i) + '.pkl', 'rb') as f:
                        self.data.extend(pickle.load(f))
        elif self.mode == 'test':
            with open('Session' + str(test_sess) + '.pkl', 'rb') as f:
                self.data.extend(pickle.load(f))
        else:
            assert False, 'Wrong mode!'

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        waveform = self.data[index]['wav']
        label = self.data[index]['emotion']
        duration = self.data[index]['duration']
        sample = {
                'waveform': torch.from_numpy(np.ascontiguousarray(waveform)), 
                'label': torch.from_numpy(np.ascontiguousarray(label)), 
                'duration': torch.from_numpy(np.ascontiguousarray(duration))}
        return sample