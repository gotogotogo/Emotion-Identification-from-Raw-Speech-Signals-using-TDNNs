# -*- coding: utf-8 -*-
import torch 
from torch.utils.data import Dataset 
import numpy as np
import pickle
from utils.utils_wav import augment


class CustomDataset(Dataset):
    def __init__(self, raw_wav_path, mode, test_sess, duration):
        self.mode = mode
        self.duration = duration
        self.data = []
        with open(raw_wav_path, 'rb') as f:
            data_dict = pickle.load(f)
        if self.mode == 'train':
            for session in data_dict:
                if session[-1] != str(test_sess):
                    self.data.extend(data_dict[session])
        elif self.mode == 'test':
            for session in data_dict:
                if session[-1] == str(test_sess):
                    self.data.extend(data_dict[session])
        else:
            assert False, 'Wrong mode!'

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        waveform = self.data[index]['wav']
        if self.mode == 'train':
            extend_wav = augment(waveform, self.duration)
        elif self.mode == 'test':
            extend_wav = waveform.resha
        label = self.data[index]['emotion']
        duration = self.data[index]['duration']
        # sample = {
        #         'waveform': torch.from_numpy(np.ascontiguousarray(extend_wav)), 
        #         'label': torch.from_numpy(np.ascontiguousarray(label)), 
        #         'duration': torch.from_numpy(np.ascontiguousarray(duration))}
        # return sample
        return extend_wav, label, duration