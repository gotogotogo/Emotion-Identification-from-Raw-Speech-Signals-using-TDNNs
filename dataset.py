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
        self.aug_data = []
        self.aug_labels = []
        self.aug_duration = []
        if self.mode == 'train':
            for session in data_dict:
                if session[-1] != str(test_sess):
                    for wav_name in data_dict[session]:
                        self.wav_paths.append(data_dict[session][wav_name]['wav_path'])
                        self.labels.append(data_dict[session][wav_name]['emotion'])
                        self.gender_labels.append(data_dict[session][wav_name]['gender'])

            self.resample1 = Resample(16000, 16000 * 0.9)
            self.resample2 = Resample(16000, 16000 * 1.0)
            self.resample3 = Resample(16000, 16000 * 1.1)
            for i in range(len(self.wav_paths)):
                waveform, sr = torchaudio.load(self.wav_paths[i])
                self.aug_data.append(utils_wav.truncate_wav(self.resample1(waveform), sr, 8)[0])
                self.aug_labels.append(self.labels[i])
                self.aug_duration.append(1)
                self.aug_data.append(utils_wav.truncate_wav(self.resample2(waveform), sr, 8)[0])
                self.aug_labels.append(self.labels[i])
                self.aug_duration.append(1)
                self.aug_data.append(utils_wav.truncate_wav(self.resample3(waveform), sr, 8)[0])
                self.aug_labels.append(self.labels[i])
                self.aug_duration.append(1)
        elif self.mode == 'test':
            for session in data_dict:
                if session[-1] == str(test_sess):
                    for wav_name in data_dict[session]:
                        self.wav_paths.append(data_dict[session][wav_name]['wav_path'])
                        self.labels.append(data_dict[session][wav_name]['emotion'])
                        self.gender_labels.append(data_dict[session][wav_name]['gender'])
            for i in range(len(self.wav_paths)):
                waveform, sr = torchaudio.load(self.wav_paths[i])
                extend_wav, duration = utils_wav.truncate_wav(waveform, sr, 8)
                self.aug_data.append(extend_wav)
                self.aug_labels.append(self.labels[i])
                self.aug_duration.append(duration)
        else:
            assert False, 'Wrong mode!'
        
        print(len(self.labels))
        label_dict = {0: 'happy', 1: 'angry', 2: 'sad', 3: 'neutral'}
        for i in range(4):
            total = sum(np.array(self.labels) == i)
            print(label_dict[i], total)
        print(len(self.aug_labels))

    def __len__(self):
        return len(self.aug_labels)
    
    def __getitem__(self, index):
        extend_wav = self.aug_data[index]
        label = self.aug_labels[index]
        duration = self.aug_duration[index]
        sample = {
                'raw_speech': torch.from_numpy(np.ascontiguousarray(extend_wav)), 
                'labels': torch.from_numpy(np.ascontiguousarray(label)), 
                'duration': torch.from_numpy(np.ascontiguousarray(duration))}
        return sample
