# define the model paras here
import torch


class PARAS:
    SR = 16000
    N_FFT = 512
    N_MEL = 150
    SAMPLE_TIME = 1   # 1s frame
    DATASET_PATH = '../Dataset/'
    BATCH_SIZE = 16
    EPOCH_NUM = 20
    USE_CUDA = True
    CUDA = torch.cuda.is_available() and USE_CUDA
    TRAIN_DATA_PATH = '../Dataset/train.h5'
    VAL_DATA_PATH = '../Dataset/valid.h5'
    TEST_DATA_PATH = '../Dataset/test.h5'
    if CUDA:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}



