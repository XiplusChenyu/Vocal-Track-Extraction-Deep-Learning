import time
import json
import numpy as np
import torch

from config import PARAS
from torch import optim
from utils import mask_scale_loss_unet


def train(model, loader, epoch_index, optimizer, versatile=True):
    start_time = time.time()
    model = model.train()
    train_loss = 0.
    batch_num = len(loader)
    _index = 0

    for _index, data in enumerate(loader):
        spec_input, target = data['mix'], data['scale_mask']
        spec_input = spec_input.unsqueeze(1)

        if PARAS.CUDA:
            spec_input = spec_input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        predicted = model(spec_input)

        loss_value = mask_scale_loss_unet(predicted, target)

        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.data.item()

        if versatile:
            if (_index + 1) % PARAS.LOG_STEP == 0:
                elapsed = time.time() - start_time
                print('Epoch{:3d} | {:3d}/{:3d} batches | {:5.2f}ms/ batch | LOSS: {:5.4f} |'
                      .format(epoch_index, _index + 1, batch_num,
                              elapsed * 1000 / (_index + 1),
                              train_loss / (_index + 1),))

    train_loss /= (_index + 1)

    print('-' * 99)
    print('End of training epoch {:3d} | time: {:5.2f}s | LOSS: {:5.4f} |'
          .format(epoch_index, (time.time() - start_time),
                  train_loss))

    return train_loss


def validate_test(model, epoch, use_loader):
    start_time = time.time()
    model = model.eval()
    v_loss = 0.
    data_loader_use = use_loader
    _index = 0

    for _index, data in enumerate(data_loader_use):
        spec_input, target = data['mix'], data['scale_mask']
        spec_input = spec_input.unsqueeze(1)

        if PARAS.CUDA:
            spec_input = spec_input.cuda()
            target = target.cuda()

        with torch.no_grad():

            predicted = model(spec_input)
            loss_value = mask_scale_loss_unet(predicted, target)

            v_loss += loss_value.data.item()

    v_loss /= (_index + 1)

    print('End of validation epoch {:3d} | time: {:5.2f}s | LOSS: {:5.4f} |'
          .format(epoch, (time.time() - start_time),
                  v_loss))
    print('-' * 99)

    return v_loss


def main_train(model, train_loader, valid_loader, log_name, save_name,
               use_simple=PARAS.USE_SIMPLE,
               lr=PARAS.LR,
               epoch_num=PARAS.EPOCH_NUM):

    start_time = time.time()
    PARAS.LOG_STEP = len(train_loader) // 4

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    t_loss, v_loss = list(), list()
    decay_cnt = 0

    build_dict = dict()

    for epoch in range(1, epoch_num + 1):
        if PARAS.CUDA:
            model.cuda()

        train_loss = train(model, train_loader, epoch, optimizer)
        validation_loss = validate_test(model, epoch, valid_loader)

        t_loss.append(train_loss)
        v_loss.append(validation_loss)

        build_dict = {
            "train_loss": t_loss,
            "valid_loss": v_loss,
        }

        with open(PARAS.LOG_PATH + log_name, 'w+') as f:
            print("****Save {0} Epoch in {1}****".format(epoch, PARAS.LOG_PATH + log_name))
            json.dump(build_dict, f)

        if len(v_loss) > 10 and np.max(v_loss[:-8]) == v_loss[-1]:
            print("****exit in epoch {0}*****".format(epoch))
            break

        # use loss to find the best model
        if np.min(t_loss) == t_loss[-1]:
            print('***Found Best Training Model***')
        if np.min(v_loss) == v_loss[-1]:
            with open(PARAS.MODEL_SAVE_PATH + save_name, 'wb') as f:
                torch.save(model.cpu().state_dict(), f)
                print('***Best Validation Model Found and Saved***')

        print('-' * 99)

        # Use loss value for learning rate scheduling
        decay_cnt += 1

        if np.min(t_loss) not in t_loss[-3:] and decay_cnt > 2:
            scheduler.step()
            decay_cnt = 0
            print('***Learning rate decreased***')
            print('-' * 99)

    total_time = round((time.time() - start_time) / 60, 2)
    print("END TRAINING, TOTAL TIME {0}min".format(total_time))

    return build_dict



