import torch
import torch.nn as nn
import torchaudio
import numpy as np


def truncate_wav(waveform, sr, duration):
    len_wav = waveform.shape[-1]
    if len_wav <int(duration * sr):
        dummy=np.zeros((1,int(duration * sr) - len_wav))
        extened_wav = np.concatenate((waveform,dummy), axis = 1)
    else:
        extened_wav = waveform[:, :int(duration * sr)]
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

