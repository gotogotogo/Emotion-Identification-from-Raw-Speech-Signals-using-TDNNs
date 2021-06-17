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
        if self.mode == 'train':
            with open('augment_wav_files_pkl', 'rb') as f:
                self.data = pickle.load(f)
            
        elif self.mode == 'test':
            self.data = []
            with open(wav_files_pkl, 'rb') as f:
                data_dict = pickle.load(f)
            for session in data_dict:
                if session[-1] != str(test_sess):
                    for wav_name in data_dict[session]:
                        wav_path = data_dict[session][wav_name]['wav_path']
                        emotion = data_dict[session][wav_name]['emotion']
                        gender = data_dict[session][wav_name]['gender']
                        waveform, sr = torchaudio.load(wav_path)
                        extend_wav, dur = utils_wav.truncate_wav(waveform, sr, duration=8)
                        self.data.append({'wav': extend_wav, 'emotion': emotion, 'gender': gender, 'duration': dur})
        else:
            assert False, 'Wrong mode!'

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            with open(self.data[index]['wav_path'], 'rb') as f:
                extend_wav = pickle.load(f)
        elif self.mode == 'test':
            extend_wav = self.data[index]['wav']
        label = self.data[index]['emotion']
        duration = self.data[index]['gender']
        sample = {
                'raw_speech': torch.from_numpy(np.ascontiguousarray(extend_wav)), 
                'labels': torch.from_numpy(np.ascontiguousarray(label)), 
                'duration': torch.from_numpy(np.ascontiguousarray(duration))}
        return sample
