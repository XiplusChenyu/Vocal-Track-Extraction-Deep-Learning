# define the model paras here
import torch


class PARAS:
    SR = 16000
    N_FFT = 512
    N_MEL = 128
    SAMPLE_TIME = 1   # 1s frame

    DATASET_PATH = '../Dataset/'

    MODEL_SAVE_PATH = '../Model/'
    LOG_PATH = '../Log/'
    E_DIM = 20

    USE_SIMPLE = False  # which is always false the simple one should be removed

    BATCH_SIZE = 32
    EPOCH_NUM = 50
    LR = 1e-5  # learning rate
    HS = 500  # DC Model Hidden size

    USE_CUDA = True
    CUDA = torch.cuda.is_available() and USE_CUDA
    if CUDA:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}

    LOG_STEP = None




