import time
import json
import numpy as np
import torch

from config import PARAS
from torch import optim
from utils import loss_function, mask_scale_loss


def train(model, loader, epoch_index, optimizer_dc, optimizer_mask, versatile=True):
    start_time = time.time()
    model = model.train()
    train_loss_dc = 0.
    train_loss_mask = 0.

    batch_num = len(loader)
    _index = 0

    for _index, data in enumerate(loader):
        spec_input, target_dc, target_mask = data['mix'], data['binary_mask'], data['scale_mask']
        shape = spec_input.size()
        target_dc = target_dc.view((shape[0], shape[1] * shape[2], -1))

        if PARAS.CUDA:
            spec_input = spec_input.cuda()
            target_dc = target_dc.cuda()
            target_mask = target_mask.cuda()

        optimizer_dc.zero_grad()
        optimizer_mask.zero_grad()

        predicted_dc, predicted_mask = model(spec_input)

        loss_value_mask = mask_scale_loss(predicted_mask, target_mask)
        loss_value_dc = loss_function(predicted_dc, target_dc)

        loss_value_mask.backward(retain_graph=True)
        loss_value_dc.backward()
        optimizer_dc.step()
        optimizer_mask.step()

        train_loss_dc += loss_value_dc.data.item()
        train_loss_mask += loss_value_mask.data.item()

        if versatile:
            if (_index + 1) % PARAS.LOG_STEP == 0:
                elapsed = time.time() - start_time
                print('Epoch{:3d} | {:3d}/{:3d} batches | {:5.2f}ms/ batch | LOSS_DC: {:5.4f} | LOSS_MASK: {:5.4f} |'
                      .format(epoch_index, _index + 1, batch_num,
                              elapsed * 1000 / (_index + 1),
                              train_loss_dc / (_index + 1),
                              train_loss_mask / (_index + 1)))

    train_loss_dc /= (_index + 1)
    train_loss_mask /= (_index + 1)

    print('-' * 99)
    print('End of training epoch {:3d} | time: {:5.2f}s |  LOSS_DC: {:5.4f} | LOSS_MASK: {:5.4f} |'
          .format(epoch_index, (time.time() - start_time),
                  train_loss_dc,
                  train_loss_mask))

    return train_loss_dc, train_loss_mask


def validate_test(model, epoch, use_loader):
    start_time = time.time()
    model = model.eval()
    v_loss_dc, v_loss_mask = 0., 0.
    data_loader_use = use_loader
    _index = 0

    for _index, data in enumerate(data_loader_use):
        spec_input, target_dc, target_mask = data['mix'], data['binary_mask'], data['scale_mask']
        shape = spec_input.size()
        target_dc = target_dc.view((shape[0], shape[1] * shape[2], -1))

        if PARAS.CUDA:
            spec_input = spec_input.cuda()
            target_dc = target_dc.cuda()
            target_mask = target_mask.cuda()

        with torch.no_grad():

            predicted_dc, predicted_mask = model(spec_input)

            loss_value_mask = mask_scale_loss(predicted_mask, target_mask)
            loss_value_dc = loss_function(predicted_dc, target_dc)

            v_loss_dc += loss_value_dc.data.item()
            v_loss_mask += loss_value_mask.data.item()

    v_loss_dc /= (_index + 1)
    v_loss_mask /= (_index + 1)

    print('End of validation epoch {:3d} | time: {:5.2f}s | LOSS_DC: {:5.4f} | LOSS_MASK: {:5.4f}'
          .format(epoch, (time.time() - start_time),
                  v_loss_dc,
                  v_loss_mask))
    print('-' * 99)

    return v_loss_dc, v_loss_mask


def main_train(model, train_loader, valid_loader, log_name, save_name,
               lr1=PARAS.LR, lr2=PARAS.LR,
               epoch_num=PARAS.EPOCH_NUM):

    start_time = time.time()
    PARAS.LOG_STEP = len(train_loader) // 4

    optimizer_dc = torch.optim.RMSprop(model.parameters(), lr=lr1)
    optimizer_mask = torch.optim.RMSprop(model.parameters(), lr=lr2)
    scheduler_dc = optim.lr_scheduler.ExponentialLR(optimizer_dc, gamma=0.5)
    scheduler_mask = optim.lr_scheduler.ExponentialLR(optimizer_mask, gamma=0.5)

    td_loss, tm_loss, vd_loss, vm_loss = list(), list(), list(), list()
    decay_cnt_dc,  decay_cnt_mask = 0, 0

    build_dict = dict()

    for epoch in range(1, epoch_num + 1):
        if PARAS.CUDA:
            model.cuda()

        train_loss = train(model, train_loader, epoch, optimizer_dc, optimizer_mask)  # a tuple here
        validation_loss = validate_test(model, epoch, valid_loader)

        td_loss.append(train_loss[0])
        tm_loss.append(train_loss[1])
        vd_loss.append(validation_loss[0])
        vm_loss.append(validation_loss[1])

        build_dict = {
            "train_loss_dc": td_loss,
            "valid_loss_dc": vd_loss,
            "train_loss_mask": tm_loss,
            "valid_loss_mask": vm_loss,
        }

        with open(PARAS.LOG_PATH + log_name, 'w+') as f:
            print("****Save {0} Epoch in {1}****".format(epoch, PARAS.LOG_PATH + log_name))
            json.dump(build_dict, f)

        if len(vm_loss) > 10 and np.max(vm_loss[:-8]) == vm_loss[-1] and np.max(vd_loss[:-8]) == vd_loss[-1]:
            print("****exit in epoch {0}*****".format(epoch))
            break

        # use loss to find the best model
        if np.min(tm_loss) == tm_loss[-1] and np.min(td_loss) == td_loss[-1]:
            print('***Found Best Training Model***')
        if np.min(vm_loss) == vm_loss[-1] and np.min(vd_loss) == vd_loss[-1]:
            with open(PARAS.MODEL_SAVE_PATH + save_name, 'wb') as f:
                torch.save(model.cpu().state_dict(), f)
                print('***Best Validation Model Found and Saved***')

        print('-' * 99)

        # Use loss value for learning rate scheduling
        decay_cnt_dc += 1
        decay_cnt_mask += 1

        if np.min(td_loss) not in td_loss[-3:] and decay_cnt_dc > 2:
            scheduler_dc.step()
            decay_cnt_dc = 0
            print('***Learning rate decreased DC***')
            print('-' * 99)

        if np.min(tm_loss) not in tm_loss[-3:] and decay_cnt_mask > 2:
            scheduler_mask.step()
            decay_cnt_mask = 0
            print('***Learning rate decreased DC***')
            print('-' * 99)

    total_time = round((time.time() - start_time) / 60, 2)
    print("END TRAINING, TOTAL TIME {0}min".format(total_time))

    return build_dict



