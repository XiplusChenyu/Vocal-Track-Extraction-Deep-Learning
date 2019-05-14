import torch.nn as nn
import torch
from data_loader import torch_dataset_loader
from config import PARAS
torch.manual_seed(1)


class HModel(nn.Module):
    def __init__(self, feature=PARAS.N_MEL, hidden_size=PARAS.HS, embedding_dim=PARAS.E_DIM):
        super(HModel, self).__init__()
        self.embedding_dim = embedding_dim

        self.gru = nn.GRU(input_size=feature,
                          hidden_size=hidden_size,
                          num_layers=4,
                          batch_first=True,
                          dropout=0.5,
                          bidirectional=True)

        self.embedding = nn.Linear(
            hidden_size * 2,
            PARAS.N_MEL * embedding_dim,
        )

        # this is for network1
        self.activation = nn.Tanh()

        # this is for network2
        self.fcBlock = nn.Sequential(nn.Dropout(0.5),
                                     nn.Linear(in_features=PARAS.N_MEL * embedding_dim, out_features=PARAS.N_MEL*2))
        self.outMask = nn.Sequential(nn.Softmax(dim=-1))

    @staticmethod
    def l2_normalize(x, dim=0, eps=1e-12):
        assert (dim < x.dim())
        norm = torch.norm(x, 2, dim, keepdim=True)
        return x / (norm + eps)

    def forward(self, inp):
        # batch, seq, feature
        n, t, f = inp.size()
        out, _ = self.gru(inp)
        out = self.embedding(out)

        out1 = self.activation(out)
        out1 = out1.view(n, -1, self.embedding_dim)
        # batch, TF, embedding
        # normalization over embedding dim
        out1 = self.l2_normalize(out1, -1)

        out2 = self.fcBlock(out)
        out2 = out2.view(n, PARAS.N_MEL, PARAS.N_MEL, 2)
        out2 = self.outMask(out2)
        return out1, out2


H_model = HModel()

if __name__ == '__main__':
    test_loader = torch_dataset_loader(PARAS.DATASET_PATH + 'test.h5', PARAS.BATCH_SIZE, True, PARAS.kwargs)
    for _index, data in enumerate(test_loader):
        spec_input = data['mix']
        label = data['binary_mask']

        if PARAS.CUDA:
            spec_input = spec_input.cuda()
            label = label.cuda()

        with torch.no_grad():

            predicted = H_model(spec_input)
            print(predicted[1])
            shape = label.size()
            label = label.view((shape[0], shape[1]*shape[2], -1))
        break
