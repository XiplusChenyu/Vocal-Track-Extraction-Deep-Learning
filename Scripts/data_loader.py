import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from config import PARAS


class TorchData(Dataset):

    def __init__(self, dataset_path):
        """
        Take the h5py dataset
        """
        super(TorchData, self).__init__()
        self.dataset = h5py.File(dataset_path, 'r')
        self.mix = self.dataset['mix']
        self.bg = self.dataset['bg']
        self.vocal = self.dataset['vocal']
        self.len = self.mix.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mix = self.mix[index].astype(np.float32)
        mix = np.reshape(mix, (1, mix.shape[0], mix.shape[1]))
        mix = torch.from_numpy(mix)

        bg = self.bg[index].astype(np.float32)
        bg = np.reshape(bg, (1, bg.shape[0], bg.shape[1]))
        bg = torch.from_numpy(bg)

        vocal = self.vocal[index].astype(np.float32)
        vocal = np.reshape(vocal, (1, vocal.shape[0], vocal.shape[1]))
        vocal = torch.from_numpy(vocal)

        sample = {
            'vocal': vocal,
            'bg': bg,
            'mix': mix,
        }

        return sample


# define the data loaders
def torch_dataset_loader(dataset, batch_size, shuffle, kwargs):
    """
    take the h5py dataset
    """
    loader = DataLoader(TorchData(dataset),
                        batch_size=batch_size,
                        shuffle=shuffle,
                        **kwargs)
    return loader


train_loader = torch_dataset_loader(PARAS.TRAIN_DATA_PATH, PARAS.BATCH_SIZE, True, PARAS.kwargs)
validation_loader = torch_dataset_loader(PARAS.VAL_DATA_PATH, PARAS.BATCH_SIZE, False, PARAS.kwargs)
test_loader = torch_dataset_loader(PARAS.TEST_DATA_PATH, PARAS.BATCH_SIZE, False, PARAS.kwargs)


if __name__ == '__main__':

    for index, data_item in enumerate(train_loader):
        print(data_item['vocal'].shape)
        print(data_item['mix'].shape)
        print(data_item['mix'].shape)
        break
