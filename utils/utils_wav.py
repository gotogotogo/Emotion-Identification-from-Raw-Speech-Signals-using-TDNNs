import torch
import torch.nn as nn
import torchaudio
import numpy as np
import random

def get_random_index(max_num, length):
    indexes = random.sample(range(0, max_num), length)
    indexes.sort()
    return indexes

def truncate_wav(waveform, sr, duration):
    len_wav = waveform.shape[-1]
    standard_len = int(duration * sr)
    if len_wav < standard_len:
        times = standard_len // len_wav
        waveform = waveform.repeat(1, times)
        len_wav = len_wav * times
        dummy=np.zeros((1,standard_len - len_wav))
        extened_wav = np.concatenate((waveform,dummy), axis = 1)
    elif len_wav > standard_len:
        # extened_wav = waveform[:, :int(duration * sr)]
        indexes = get_random_index(len_wav, standard_len)
        extend_wav = waveform[:, indexes]
    else:
        extend_wav = waveform
        
    return extened_wav

def speed_perturbation():
    pass 

def speech_collate(batch):
    targets = []
    raw_data = []
    for sample in batch:
        raw_data.append(sample['raw_speech'])
        targets.append((sample['labels']))
    return raw_data, targets

