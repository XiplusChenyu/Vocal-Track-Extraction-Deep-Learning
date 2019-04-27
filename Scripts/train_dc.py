import time
import json
import numpy as np
from cluster_model import D_model
import torch
from torch import optim
from utils import loss_function_dc
from config import PARAS
from data_loader import train_loader, validation_loader, test_loader

PARAS.LOG_STEP = len(train_loader) // 16

optimizer = torch.optim.RMSprop(D_model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)


def train(model, epoch, versatile=True):
    start_time = time.time()
    model = model.train()
    train_loss = 0.
    batch_num = len(train_loader)
    _index = 0

    for _index, data in enumerate(train_loader):
        spec_input, target = data['mel'], data['target']

        if PARAS.CUDA:
            spec_input = spec_input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        predicted = model(spec_input)

        loss_value = loss_function_dc(predicted, target)

        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.data.item()

        if versatile:
            if (_index + 1) % PARAS.LOG_STEP == 0:
                elapsed = time.time() - start_time
                print('Epoch{:3d} | {:3d}/{:3d} batches | {:5.2f}ms/ batch | LOSS: {:5.4f} |'
                      .format(epoch, _index + 1, batch_num,
                              elapsed * 1000 / (_index + 1),
                              train_loss / (_index + 1),))

    train_loss /= (_index + 1)

    print('-' * 99)
    print('End of training epoch {:3d} | time: {:5.2f}s | LOSS: {:5.4f} |'
          .format(epoch, (time.time() - start_time),
                  train_loss))

    return train_loss


def validate_test(model, epoch, test=False):
    start_time = time.time()
    model = model.eval()
    v_loss = 0.
    data_loader_use = validation_loader if not test else test_loader
    _index = 0
    for _index, data in enumerate(data_loader_use):
        spec_input, target = data['mel'], data['target']

        if PARAS.CUDA:
            spec_input = spec_input.cuda()
            target = target.cuda()

        with torch.no_grad():

            predicted = model(spec_input)

            loss_value = loss_function_dc(predicted, target)

            v_loss += loss_value.data.item()

    v_loss /= (_index + 1)

    print('End of validation epoch {:3d} | time: {:5.2f}s | LOSS: {:5.4f} |'
          .format(epoch, (time.time() - start_time),
                  v_loss))
    print('-' * 99)

    return v_loss


if __name__ == '__main__':
    t_loss, v_loss = [], []
    decay_cnt = 0

    for epoch in range(1, PARAS.EPOCH_NUM + 1):
        if PARAS.CUDA:
            D_model.cuda()

        train_loss = train(D_model, epoch)
        validation_loss = validate_test(D_model, epoch, test=False)

        t_loss.append(train_loss)
        v_loss.append(validation_loss)

        # use loss to find the best model
        if np.max(t_loss) == t_loss[-1]:
            print('***Found Best Training Model***')
        if np.max(v_loss) == v_loss[-1]:
            with open(PARAS.MODEL_SAVE_PATH_1, 'wb') as f:
                torch.save(D_model.cpu().state_dict(), f)
                print('***Best Validation Model Found and Saved***')

        print('-' * 99)

        # Use BCE loss value for learning rate scheduling
        decay_cnt += 1

        if np.min(t_loss) not in t_loss[-3:] and decay_cnt > 2:
            scheduler.step()
            decay_cnt = 0
            print('***Learning rate decreased***')
            print('-' * 99)

    with open(PARAS.TEST_DATA_PATH, 'w+') as t, open(PARAS.VAL_DATA_PATH, 'w+') as v:
        json.dump(t_loss, t)
        json.dump(v_loss, v)
