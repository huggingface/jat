import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_shape, out_channels):
        super().__init__()
        out_shape = (out_channels, in_shape[1], in_shape[2])

        self.conv1 = nn.Conv2d(in_shape[0], out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.norm1 = nn.LayerNorm(out_shape)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.norm2 = nn.LayerNorm(out_shape)

        # Handling the change in dimensions with a 1x1 convolution
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_shape[0], out_channels, kernel_size=1, stride=1), nn.LayerNorm(out_shape)
        )

    def forward(self, x):
        out = F.leaky_relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        return F.leaky_relu(out)


class AttentionLayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 8, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImageEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)  # 42x42
        self.norm1 = nn.InstanceNorm2d(32)
        self.att1 = AttentionLayer(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 21x21
        self.norm2 = nn.InstanceNorm2d(64)
        self.att2 = AttentionLayer(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 11x11
        self.norm3 = nn.InstanceNorm2d(128)
        self.att3 = AttentionLayer(128)
        self.fc = nn.Linear(128 * 11 * 11, hidden_size)  # Adjusted to the new spatial dimension

    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.conv1(x)))
        x = self.att1(x)
        x = F.leaky_relu(self.norm2(self.conv2(x)))
        x = self.att2(x)
        x = F.leaky_relu(self.norm3(self.conv3(x)))
        x = self.att3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc_dec = nn.Linear(hidden_size, 10 * 10 * 128)
        self.trans_conv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock((128, 21, 21), 64)
        self.trans_conv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock((64, 42, 42), 32)
        self.trans_conv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc_dec(x)
        x = x.view(x.size(0), 128, 10, 10)
        x = self.trans_conv1(x)
        x = F.interpolate(x, size=(21, 21))
        x = self.dec1(x)
        x = self.trans_conv2(x)
        x = self.dec2(x)
        x = self.trans_conv3(x)
        x = torch.tanh(self.dec3(x))
        return x
