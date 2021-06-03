# -*- coding: utf-8 -*-
import torch 
from torch.utils.data import Dataset 
import torchaudio  
from torchaudio.transforms import Resample

import numpy as np
import pickle
from utils import utils_wav


class CustomDataset(Dataset):
    def __init__(self, wav_files_pkl, mode, test_sess):
        self.mode = mode
        with open(wav_files_pkl, 'rb') as f:
            data_dict = pickle.load(f)
        self.wav_paths = []
        self.labels = []
        self.gender_labels = []
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
        self.aug_data = []
        self.aug_label = []
        self.resample1 = Resample(16000, 16000 * 0.9)
        self.resample2 = Resample(16000, 16000 * 1.0)
        self.resample3 = Resample(16000, 16000 * 1.1)
        for i in range(len(self.wav_paths)):
            waveform, sr = torchaudio.load(self.wav_paths[i])
            self.aug_data.append(utils_wav.truncate_wav(self.resample1(waveform), sr, 8))
            self.aug_label.append(self.labels[i])
            self.aug_data.append(utils_wav.truncate_wav(self.resample2(waveform), sr, 8))
            self.aug_label.append(self.labels[i])
            self.aug_data.append(utils_wav.truncate_wav(self.resample3(waveform), sr, 8))
            self.aug_label.append(self.labels[i])
        print(len(self.labels))
        print(len(self.aug_label))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        # wav_path = self.wav_paths[index]
        # label = self.labels[index]
        # #gender = self.gender_labels[index]
        # extend_wav = utils_wav.load_wav(wav_path,min_dur_sec=8)
        # sample = {'raw_speech': torch.from_numpy(np.ascontiguousarray(extend_wav)), 'labels': torch.from_numpy(np.ascontiguousarray(label))}
        # return sample
        extend_wav = self.aug_data[index]
        label = self.aug_label[index]
        sample = {'raw_speech': torch.from_numpy(np.ascontiguousarray(extend_wav)), 'labels': torch.from_numpy(np.ascontiguousarray(label))}
        return sample