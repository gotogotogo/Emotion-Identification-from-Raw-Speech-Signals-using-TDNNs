import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import Resample
import numpy as np
import random

def get_random_index(max_num, length):
    indexes = random.sample(range(0, max_num), length)
    indexes.sort()
    return indexes

def truncate(waveform, duration, sr=16000):
    '''
        truncate waveform to the specified time duration 
        return :
            extend_wav: truncated waveform
    '''
    cur_len = waveform.shape[-1]
    standard_len = int(duration * sr)
    if cur_len < standard_len:
        dummy = np.zeros((1, standard_len - cur_len))
        extend_wav = np.concatenate((waveform, dummy), axis = 1)
    elif cur_len > standard_len:
        # extend_wav = waveform[:, :int(duration * sr)]
        indexes = get_random_index(cur_len, standard_len)
        extend_wav = waveform[:, indexes]
    else:
        extend_wav = waveform

    return extend_wav

def resample(waveform):
    rate = 1 + (random.randint(0, 2) - 1) * 0.1
    resample = Resample(16000, 16000 * rate)
    return resample(waveform) 

def amplitude_modulate(waveform, min_gain_db=-12, max_gain_db=12):
    amplitude = random.uniform(min_gain_db, max_gain_db)
    amplitude = 10 ** (amplitude / 20)
    return waveform * amplitude

def augment(waveform, duration):
    extend_wav = resample(waveform)
    extend_wav = amplitude_modulate(extend_wav)
    extend_wav = truncate(extend_wav, duration)
    return extend_wav

def speech_collate(batch):
    targets = []
    raw_data = []
    duration = []
    vad = []
    for sample in batch:
        raw_data.append(sample['waveform'])
        targets.append(sample['label'])
        duration.append(sample['duration'])
        vad.append(sample['vad'])
    return raw_data, targets, duration, vad