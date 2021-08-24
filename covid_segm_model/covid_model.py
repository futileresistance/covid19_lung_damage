import torch
import torch.nn as nn
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, channel1, channel2, kernel_size=(3, 3), padding=1):
        super(StackEncoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ConvBlock(channel1, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
        )

    def forward(self, x):
        copy_out = self.block(x)
        poolout = self.maxpool(copy_out)
        return copy_out, poolout


class StackDecoder(nn.Module):
    def __init__(self, copy_channel, channel1, channel2, kernel_size=(3, 3), padding=1):
        super(StackDecoder, self).__init__()
        self.unConv = nn.ConvTranspose2d(channel1, channel1, kernel_size=(2, 2), stride=2)
        self.block = nn.Sequential(
            ConvBlock(channel1 + copy_channel, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
        )

    def forward(self, x, down_copy):
        _, channels, height, width = down_copy.size()
        x = self.unConv(x)
        x = torch.cat([x, down_copy], 1)
        x = self.block(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.down1 = StackEncoder(1, 32, kernel_size=(3, 3))
        self.down2 = StackEncoder(32, 64, kernel_size=(3, 3))
        self.down3 = StackEncoder(64, 128, kernel_size=(3, 3))
        self.down4 = StackEncoder(128, 256, kernel_size=(3, 3))

        self.center = ConvBlock(256, 256, kernel_size=(3, 3), padding=1)

        self.up4 = StackDecoder(256, 256, 128, kernel_size=(3, 3))
        self.up3 = StackDecoder(128, 128, 64, kernel_size=(3, 3))
        self.up2 = StackDecoder(64, 64, 32, kernel_size=(3, 3))
        self.up1 = StackDecoder(32, 32, 16, kernel_size=(3, 3))
        self.conv = nn.Conv2d(16, 1, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        copy1, out = self.down1(x)
        copy2, out = self.down2(out)
        copy3, out = self.down3(out)
        copy4, out = self.down4(out)

        out = self.center(out)

        up4 = self.up4(out, copy4)
        up3 = self.up3(up4, copy3)
        up2 = self.up2(up3, copy2)
        up1 = self.up1(up2, copy1)

        out = self.conv(up1)
        out = nn.Sigmoid()(out)

        return out

model = Unet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('covid_segm_model/lungNET_take3', map_location=device))
model.eval()
print('covid model loaded...')