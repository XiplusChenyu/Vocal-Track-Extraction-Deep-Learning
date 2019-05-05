# define the model paras here
import torch


class PARAS:
    SR = 16000
    N_FFT = 512
    N_MEL = 150
    SAMPLE_TIME = 1   # 1s frame

    DATASET_PATH = '../Dataset/'
    TRAIN_DATA_PATH = '../Dataset/train.h5'
    VAL_DATA_PATH = '../Dataset/valid.h5'
    TEST_DATA_PATH = '../Dataset/test.h5'

    MODEL_SAVE_PATH_1 = '../Model/dc_model.h5'
    T_LOG_PATH = '../Log/train_loss.json'
    V_LOG_PATH = '../Log/valid_loss.json'

    BATCH_SIZE = 6
    EPOCH_NUM = 100

    USE_CUDA = True
    CUDA = torch.cuda.is_available() and USE_CUDA
    if CUDA:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}

    LOG_STEP = None




