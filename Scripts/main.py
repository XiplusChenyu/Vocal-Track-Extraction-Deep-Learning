from cluster_model import D_model
from dc_train import main_train
from config import PARAS
from data_loader import torch_dataset_loader

train_file = 'train.h5'
valid_file = 'valid.h5'
train_loader = torch_dataset_loader(PARAS.DATASET_PATH + train_file, PARAS.BATCH_SIZE, True, PARAS.kwargs)
valid_loader = torch_dataset_loader(PARAS.DATASET_PATH + valid_file, PARAS.BATCH_SIZE, True, PARAS.kwargs)
if __name__ == '__main__':
    res = main_train(model=D_model,
                     train_loader=train_loader,
                     valid_loader=valid_loader,
                     log_name='dc_model_may_6.json',
                     save_name='dc_model_may_6.pt')
