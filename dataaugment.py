import pickle
import torchaudio
from torchaudio.transforms import Resample
import random
from utils import utils_wav
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--data_path', type=str, default='data/')
def augment(wav_files_pkl, data_path, duration=8):
    with open(wav_files_pkl, 'rb') as f:
        data_dict = pickle.load(f)
    data_aug = []
    for session in tqdm(data_dict):
        if session[-1] != '5':
            for wav_name in tqdm(data_dict[session]):
                waveform, sr = torchaudio.load(data_dict[session][wav_name]['wav_path'])
                resamples = resample(waveform)
                for i in range(len(resamples)):
                    amplitudes = amplitude_modulate(resamples[i])
                    for j in range(len(amplitudes)):
                        extend_wav, dur = utils_wav.truncate_wav(amplitudes[j], sr, duration=duration)
                        path = data_path + wav_name + '_' + str(i) + '_' + str(j) + '.pkl'
                        emotion = data_dict[session][wav_name]['emotion']
                        gender = data_dict[session][wav_name]['gender']
                        with open(path, 'wb') as f:
                            pickle.dump(extend_wav, f)
                        data_aug.append({'wav_path': path, 'emotion': emotion, 'gender': gender, 'duration': dur})
    with open('augment_wav_files_pkl', 'wb') as f:
            pickle.dump(data_aug, f)

def resample(waveform):
    resample1 = Resample(16000, 16000 * 0.9)
    resample2 = Resample(16000, 16000 * 1.0)
    resample3 = Resample(16000, 16000 * 1.1)
    result = []
    result.append(resample1(waveform))
    result.append(resample2(waveform))
    result.append(resample3(waveform))
    return result 

def amplitude_modulate(waveform, min_gain_db=-12, max_gain_db=12, size=10):
    result = []
    for i in range(size):
        amplitude = random.uniform(min_gain_db, max_gain_db)
        amplitude = 10 ** (amplitude / 20)
        result.append(waveform * amplitude)
    return result

if __name__ == "__main__":
    args = parser.parse_args()
    augment('wav_collect_files.pkl', args.data_path)