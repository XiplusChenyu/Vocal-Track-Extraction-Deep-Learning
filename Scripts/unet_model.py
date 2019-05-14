import torch
import torch.nn as nn
import torch.nn.functional as F
from hybrid_model import HModel as Cluster
from config import PARAS
"""
Thanks https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py for his model structure
"""


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convBlock(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.inConvBlock = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.inConvBlock(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.mpConv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpConv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(UpConv, self).__init__()
        if bilinear:  # try to also train the
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        yd = x2.size()[2] - x1.size()[2]
        xd = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (xd // 2, xd - xd // 2,
                        yd // 2, yd - yd // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# full assembly of the sub-parts to form the complete net
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.inc = InConv(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DoubleConv(512, 512)
        self.up1 = UpConv(1024, 256)
        self.up2 = UpConv(512, 128)
        self.up3 = UpConv(256, 64)
        self.up4 = UpConv(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x.squeeze(1)  # make single channel output


# full assembly of the sub-parts to form the complete net
class UNetEnhanced(nn.Module):
    def __init__(self, reference_model_name, n_channels=3, n_classes=1):
        super(UNetEnhanced, self).__init__()
        self.reference_model = Cluster()
        self.reference_model.load_state_dict(torch.load(PARAS.MODEL_SAVE_PATH+reference_model_name,
                                                        map_location='cpu'))
        self.inc = InConv(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DoubleConv(512, 512)
        self.up1 = UpConv(1024, 256)
        self.up2 = UpConv(512, 128)
        self.up3 = UpConv(256, 64)
        self.up4 = UpConv(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Dealer
        c = x.squeeze(1)
        _, c1 = self.reference_model(c)
        c2 = c1.view((-1, 2, PARAS.N_MEL, PARAS.N_MEL))
        x = torch.cat((c2, x), dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x.squeeze(1)  # make single channel output


U_model = UNet()  # The model also use bce loss for train


if __name__ == '__main__':
    UH_model = UNetEnhanced('hd_model_may_11.pt')
    # from torchsummary import summary
    # summary(UH_model, (1, 128, 128), batch_size=16)

    from data_loader import torch_dataset_loader
    from utils import mask_scale_loss_unet

    test_loader = torch_dataset_loader(PARAS.DATASET_PATH + 'test.h5', PARAS.BATCH_SIZE, True, PARAS.kwargs)
    for _index, data in enumerate(test_loader):
        spec_input = data['mix']
        spec_input = spec_input.unsqueeze(1)
        label = data['scale_mask']
        UH_model.eval()

        if PARAS.CUDA:
            spec_input = spec_input.cuda()
            label = label.cuda()

        with torch.no_grad():

            predicted = UH_model(spec_input)
            print(mask_scale_loss_unet(predicted, label))
        break
