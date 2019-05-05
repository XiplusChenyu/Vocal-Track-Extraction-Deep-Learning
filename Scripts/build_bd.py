import numpy as np
import librosa
import os
import h5py
import time
import random
from config import PARAS
from mel_dealer import mel_converter

"""
Run this file to generate dbs in background
"""

MIX_PATH = '../DSD100/Mixtures'
SRC_PATH = '../DSD100/Sources'

dev_file_paths = dict()
test_file_paths = dict()

for (dirpath, dirnames, filenames) in os.walk(MIX_PATH):
    if not dirnames:
        if 'Test' in dirpath:
            file_name = dirpath.split('Test/')[-1].split('-')[0].strip()
            test_file_paths[file_name] = dict()
            test_file_paths[file_name]['mix'] = dirpath + '/' + filenames[0]
        elif 'Dev' in dirpath:
            file_name = dirpath.split('Dev/')[-1].split('-')[0].strip()
            dev_file_paths[file_name] = dict()
            dev_file_paths[file_name]['mix'] = dirpath + '/' + filenames[0]

for (dirpath, dirnames, filenames) in os.walk(SRC_PATH):
    if not dirnames:
        if 'Test' in dirpath:
            file_name = dirpath.split('Test/')[-1].split('-')[0].strip()
            for trackname in filenames:
                track = trackname.split('.wav')[0]
                test_file_paths[file_name][track] = dirpath + '/' + trackname
        elif 'Dev' in dirpath:
            file_name = dirpath.split('Dev/')[-1].split('-')[0].strip()
            for trackname in filenames:
                track = trackname.split('.wav')[0]
                dev_file_paths[file_name][track] = dirpath + '/' + trackname


def remove_silence(vocal, target_track):
    """
    This function remove slience part
    notice it's not an in place dealer
    """
    remove_list = list()
    for i, point in enumerate(vocal):
        if 10**-4 > abs(point): # be careful of the silence part shreshold
            remove_list.append(i)
    vocal_out = np.delete(vocal, remove_list)
    target_track_out = np.delete(target_track, remove_list)
    return vocal_out, target_track_out


def sound_tracks_extractor(file_path):
    """
    Take in a file path dictionary, return:
    (Normalized track)
    vocal track as target
    mixed track as input
    """
    signals = dict()
    for key in file_path:
        if key == 'mix':
            continue
        signals[key], _ = librosa.load(file_path.get(key), sr=PARAS.SR)
        signals[key] = librosa.util.normalize(signals.get(key))

    vocal_track = signals.get('vocals')
    return_vocal, _ = remove_silence(vocal_track, vocal_track)
    background_track = np.zeros(len(return_vocal))

    for key in signals:
        if 'vocals' == key:
            continue
        _, signals[key] = remove_silence(vocal_track, signals.get(key))
        background_track = signals.get(key) if not len(background_track) else background_track + signals.get(key)

    mix_track = background_track + return_vocal

    return return_vocal, background_track, mix_track


def frame_feature_extractor(signal, mel_converter=mel_converter):
    """
    Takes in new signals and create mel chunks
    """
    S = mel_converter.signal_to_melspec(signal)
    if not S.shape[0] % PARAS.N_MEL == 0:
        S = S[:-1 * (S.shape[0] % PARAS.N_MEL)]  # divide the mel spectrogram

    chunk_num = int(S.shape[0] / PARAS.N_MEL)
    mel_chunks = np.split(S, chunk_num)  # create PARAS.N_MEL * PARAS.N_MEL data frames
    return mel_chunks, chunk_num


file_paths = dict(list(dev_file_paths.items())
                  + list(test_file_paths.items()))

print("start loading data...")
file_count = 100
chunk_count = 0

for key, file_dict in file_paths.items():
    ss_time = time.time()
    count = 0
    vocal_signal, bg_signal, mix_signal = sound_tracks_extractor(file_dict)
    vocal_chunks, cn = frame_feature_extractor(vocal_signal)
    mix_chunks, _ = frame_feature_extractor(mix_signal)
    bg_chunks, _ = frame_feature_extractor(bg_signal)

    c_setpath = PARAS.DATASET_PATH + '{0}.h5'.format(key)
    c_dataset = h5py.File(c_setpath, 'a')

    c_dataset.create_dataset('mix',
                             shape=(cn, PARAS.N_MEL, PARAS.N_MEL),
                             dtype=np.float32)

    c_dataset.create_dataset('bg',
                             shape=(cn, PARAS.N_MEL, PARAS.N_MEL),
                             dtype=np.float32)

    c_dataset.create_dataset('vocal',
                             shape=(cn, PARAS.N_MEL, PARAS.N_MEL),
                             dtype=np.float32)

    for idx in range(cn):
        c_dataset['vocal'][count] = vocal_chunks[idx]
        c_dataset['mix'][count] = mix_chunks[idx]
        c_dataset['bg'][count] = bg_chunks[idx]
        count += 1
        chunk_count += 1

    file_count -= 1
    epoch_s = round(time.time() - ss_time, 2)
    print("=>${0}[{1}s|LEFT:{2}|CHUNK:{3}] ".format(key, epoch_s, file_count, chunk_count), end="\r", flush=True)
    c_dataset.close()
print("TOTAL CHUNK: {0}".format(chunk_count))


all_setpath = PARAS.DATASET_PATH + 'all.h5'
all_dataset = h5py.File(all_setpath, 'a')

all_dataset.create_dataset('mix',
                           shape=(chunk_count, PARAS.N_MEL, PARAS.N_MEL),
                           dtype=np.float32)

all_dataset.create_dataset('vocal',
                           shape=(chunk_count, PARAS.N_MEL, PARAS.N_MEL),
                           dtype=np.float32)

all_dataset.create_dataset('bg',
                           shape=(chunk_count, PARAS.N_MEL, PARAS.N_MEL),
                           dtype=np.float32)

count = 0
for (dirpath, dirnames, filenames) in os.walk(PARAS.DATASET_PATH):
    if filenames:
        for set_name in filenames:
            if 'h5' not in set_name or 'all' in set_name:
                continue
            set_path = PARAS.DATASET_PATH + set_name
            tmp_dataset = h5py.File(set_path, 'r')
            tmp_count = tmp_dataset['vocal'].shape[0]
            all_count += tmp_count
            for i in range(tmp_count):
                all_dataset['mix'][count] = tmp_dataset['mix'][i]
                all_dataset['vocal'][count] = tmp_dataset['vocal'][i]
                all_dataset['bg'][count] = tmp_dataset['bg'][i]
                count += 1
            tmp_dataset.close()
all_dataset.close()
print("Create all data: {0}".format(count))

all_setpath = PARAS.DATASET_PATH + 'all.h5'
all_dataset = h5py.File(all_setpath, 'r')
chunk_count = 14257

train_set = PARAS.DATASET_PATH+'train.h5'
valid_set = PARAS.DATASET_PATH+'valid.h5'
test_set = PARAS.DATASET_PATH+'test.h5'

train_file = int(chunk_count * 0.8)
valid_file = int(chunk_count * 0.1)
test_file = chunk_count - train_file - valid_file

files = [int(a) for a in [train_file, valid_file, test_file]]
sets = [train_set, valid_set, test_set]

idx = [i for i in range(chunk_count)]
random.seed(516)
random.shuffle(idx)

train_idx = idx[:files[0]]
valid_idx = idx[files[0]: files[0]+files[1]]
test_idx = idx[-files[2]:]
indices = [train_idx, valid_idx, test_idx]

for i, dset in enumerate(sets):
    s_set = h5py.File(dset, 'a')
    indice = indices[i]
    file_num = files[i]

    s_set.create_dataset('vocal', shape=(file_num, PARAS.N_MEL, PARAS.N_MEL), dtype=np.float32)
    s_set.create_dataset('mix', shape=(file_num, PARAS.N_MEL, PARAS.N_MEL), dtype=np.float32)
    s_set.create_dataset('bg', shape=(file_num, PARAS.N_MEL, PARAS.N_MEL), dtype=np.float32)

    count = 0
    for i in indice:
        s_set['vocal'][count] = all_dataset['vocal'][i]
        s_set['mix'][count] = all_dataset['mix'][i]
        s_set['bg'][count] = all_dataset['bg'][i]
        count += 1

        if count % 100 == 0:
            print('=>{0}'.format(count), end="")

    s_set.close()
    print()
    print('Create Separate Datasets {0}'.format(dset))

exit(0)
