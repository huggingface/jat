import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor


class ResidualBlock(nn.Module):
    def __init__(self, in_shape, out_channels):
        super().__init__()
        out_shape = (out_channels, in_shape[1], in_shape[2])

        self.conv1 = nn.Conv2d(in_shape[0], out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(out_shape)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.LayerNorm(out_shape)

        # Handling the change in dimensions with a 1x1 convolution
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_shape[0], out_channels, kernel_size=1, stride=1), nn.LayerNorm(out_shape)
        )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ImageEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.enc1 = ResidualBlock((3, 84, 84), 32)
        self.enc2 = ResidualBlock((32, 42, 42), 64)
        self.enc3 = ResidualBlock((64, 21, 21), 128)
        self.fc_enc = nn.Linear(10 * 10 * 128, hidden_size)

    def forward(self, x):
        x = self.enc1(x)
        x = F.max_pool2d(x, 2)
        x = self.enc2(x)
        x = F.max_pool2d(x, 2)
        x = self.enc3(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc_enc(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc_dec = nn.Linear(hidden_size, 10 * 10 * 128)
        self.dec1 = ResidualBlock((128, 21, 21), 64)
        self.dec2 = ResidualBlock((64, 42, 42), 32)
        self.dec3 =  nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc_dec(x)
        x = x.view(x.size(0), 128, 10, 10)
        x = F.interpolate(x, size=(21, 21))
        x = self.dec1(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.dec2(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.sigmoid(self.dec3(x))
        return x


# Instantiate the model
model = nn.Sequential(ImageEncoder(128), ImageDecoder(128))
print(model)
x = torch.rand((2, 3, 84, 84))
print(model(x).shape)
