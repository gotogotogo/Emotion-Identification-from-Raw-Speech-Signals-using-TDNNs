# -*- coding: utf-8 -*-

import os
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
    data_dict = {}
    for speaker in tqdm(os.listdir(root_path)):
        data_dict[speaker] = {}
        wav_dir =  os.path.join(root_path, speaker, 'sentences/wav')
        emo_dir = os.path.join(root_path, speaker, 'dialog/EmoEvaluation')
        for sess in os.listdir(wav_dir):
            label_list = {}
            emotxt = emo_dir + '/' + sess + '.txt'
            with open(emotxt, 'r') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if(line[0] == '['):
                        t = line.split()
                        label_list[t[3]] = t[4]
            files = glob.glob(os.path.join(wav_dir, sess, '*.wav'))
            for file_name in files:
                #print(file_name)
                wave_name = file_name.split('/')[-1][:-4]
                #print(wave_name)
                emotion = label_list[wave_name]
                if emotion in ['ang', 'sad', 'neu', 'exc', 'hap']:
                    data_dict[speaker][wave_name] = {}
                    data_dict[speaker][wave_name]['wav_path'] = file_name
                    data_dict[speaker][wave_name]['emotion'] = emotion_id[emotion]
                    data_dict[speaker][wave_name]['gender'] = gender_id[wave_name[5]]
    for s in data_dict:
        print('len of s: ', len(data_dict[s]))
    with open('wav_collect_files.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
        print('successfully collect wav files')           

def collect_durations():
    with open('wav_collect_files.pkl', 'rb') as f:         
        data_dict = pickle.load(f)
        duration_dict = {}
        duration_dict[4] = 0
        duration_dict[8] = 0
        duration_dict[12] = 0
        duration_dict[20] = 0
        for speaker in data_dict:
            for wave_name in data_dict[speaker]:
                wav, _ = torchaudio.load(wave_name)
                dur = wav.shape[1] / 16000
                if dur < 4:
                    duration_dict[4] += 1
                elif dur < 8:
                    duration_dict[8] += 1
                elif dur < 12:
                    duration_dict[12] += 1
                else:
                    duration_dict[20] += 1
        print(duration_dict)
if __name__ == "__main__":
    args = parser.parse_args()
    collect_files(args.raw_path)
    collect_durations()