# -*- coding: utf-8 -*-

import os
import numpy as np
import glob
import pickle
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--raw_path', type=str, default='F:/IEMOCAP')
emotion_id = {'hap': 0, 'exc': 0, 'ang': 1, 'sad': 2, 'neu': 3}
gender_id = {'M':0,'F':1}
def collect_files(root_path):
    data_dict = {}
    for speakerdir in tqdm(os.listdir(root_path)):
        data_dict[speakerdir] = {}
        wav_dir =  os.path.join(root_path, speakerdir, 'sentences/wav')
        emo_dir = os.path.join(root_path, speakerdir, 'dialog/EmoEvaluation')
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
                wave_name = file_name.split('\\')[-1][:-4]
                #print(wave_name)
                emotion = label_list[wave_name]
                if emotion in ['ang', 'sad', 'neu', 'exc', 'hap']:
                    data_dict[speakerdir][wave_name] = {}
                    data_dict[speakerdir][wave_name]['wav_path'] = file_name
                    data_dict[speakerdir][wave_name]['emotion'] = emotion_id[emotion]
                    data_dict[speakerdir][wave_name]['gender'] = gender_id[wave_name[5]]
    for s in data_dict:
        print('len of s: ', len(data_dict[s]))
    with open('wav_collect_files.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
        print('successfully collect wav files')           


if __name__ == "__main__":
    args = parser.parse_args()
    collect_files(args.raw_path)