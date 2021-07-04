# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import glob
import pickle
import argparse
from tqdm import tqdm
import torchaudio

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--raw_path', type=str, default='F:/IEMOCAP')
emotion_id = {'hap': 0, 'exc': 0, 'ang': 1, 'sad': 2, 'neu': 3}
gender_id = {'M':0,'F':1}
def collect_files(root_path):
    '''
    data_dict:{
        'Session1': [{'wav': wav, 'emotion': emotion, 'gender': gender, 'duration': duration, 'vad': [V, A, D]},...],
        ......
        'Session5': [{'wav': wav, 'emotion': emotion, 'gender': gender, 'duration': duration, 'vad': [V, A, D]},...],
    }
    '''
    data_dict = {}
    for speaker in tqdm(os.listdir(root_path)):
        data_dict[speaker] = []
        wav_dir =  os.path.join(root_path, speaker, 'sentences/wav')
        emo_dir = os.path.join(root_path, speaker, 'dialog/EmoEvaluation')
        for sess in os.listdir(wav_dir):
            label_list = {}
            VAD_list = {}
            emotxt = emo_dir + '/' + sess + '.txt'
            with open(emotxt, 'r') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if(line[0] == '['):
                        t = line.split()
                        V = t[5][1:-1]
                        A = t[6][0:-1]
                        D = t[7][:-1]
                        label_list[t[3]] = t[4]
                        VAD_list[t[3]] = [float(V), float(A), float(D)]
            files = glob.glob(os.path.join(wav_dir, sess, '*.wav'))
            for file_name in files:
                #print(file_name)
                wave_name = file_name.split('\\')[-1][:-4]
                waveform, _ = torchaudio.load(file_name)
                #print(wave_name)
                emotion = label_list[wave_name]
                VAD = VAD_list[wave_name]
                if emotion in ['ang', 'sad', 'neu', 'exc', 'hap']:
                    data_dict[speaker].append({'wav': waveform, 'emotion': emotion_id[emotion], 'gender': gender_id[wave_name[5]], 'duration': waveform.shape[1] / 16000, 'vad': VAD})
        print('len of ', speaker, ' :', len(data_dict[speaker]))
    with open('raw_wavs.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    print('successfully collect wav files')           


def collect_durations():
    with open('raw_wavs.pkl', 'rb') as f:         
        data_dict = pickle.load(f)
        duration_dict = {}
        duration_dict[1] = 0
        duration_dict[2] = 0
        duration_dict[4] = 0
        duration_dict[8] = 0
        duration_dict[12] = 0
        duration_dict['gt_12'] = 0
        for speaker in data_dict:
            if speaker[-1] == '5':
                for i in range(len(data_dict[speaker])):
                    dur = data_dict[speaker][i]['duration']
                    if dur < 1:
                        duration_dict[1] += 1
                    elif dur < 2:
                        duration_dict[2] += 1
                    elif dur < 4:
                        duration_dict[4] += 1
                    elif dur < 8:
                        duration_dict[8] += 1
                    elif dur < 12:
                        duration_dict[12] += 1
                    else:
                        duration_dict['gt_12'] += 1
        print(duration_dict)
if __name__ == "__main__":
    args = parser.parse_args()
    collect_files(args.raw_path)
    collect_durations()