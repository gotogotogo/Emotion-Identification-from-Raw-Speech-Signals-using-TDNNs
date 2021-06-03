# -*- coding: utf-8 -*-
import numpy as np
import pickle
import torch
from utils import utils_wav
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, wav_files_pkl, mode, test_sess):
        self.mode = mode
        with open(wav_files_pkl, 'rb') as f:
            data_dict = pickle.load(f)
        self.wav_paths = []
        self.labels = []
        self.gender_labels = []
        self.data = []
        if self.mode == 'train':
            for session in data_dict:
                if session[-1] != str(test_sess):
                    for wav_name in data_dict[session]:
                        self.wav_paths.append(data_dict[session][wav_name]['wav_path'])
                        self.labels.append(data_dict[session][wav_name]['emotion'])
                        self.gender_labels.append(data_dict[session][wav_name]['gender'])
        elif self.mode == 'test':
            for session in data_dict:
                if session[-1] == str(test_sess):
                    for wav_name in data_dict[session]:
                        self.wav_paths.append(data_dict[session][wav_name]['wav_path'])
                        self.labels.append(data_dict[session][wav_name]['emotion'])
                        self.gender_labels.append(data_dict[session][wav_name]['gender'])
        else:
            assert False, 'Wrong mode!'
        for wav_path in self.wav_paths:
            extend_wav = utils_wav.load_wav(wav_path,min_dur_sec=8)
            self.data.append(extend_wav)
        with open(self.mode + '_data.pkl', 'wb') as f:
            pickle.dump(self.data, f)
        print(len(self.labels))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        label = self.labels[index]
        #gender = self.gender_labels[index]
        extend_wav = utils_wav.load_wav(wav_path,min_dur_sec=8)
        sample = {'raw_speech': torch.from_numpy(np.ascontiguousarray(extend_wav)), 'labels': torch.from_numpy(np.ascontiguousarray(label))}
        return sample

def augment(waveform):
    pass