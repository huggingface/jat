import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride != 1 or in_channels != out_channels:
            self.shorcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels),
            )
        else:
            self.shorcut = None
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x if self.shorcut is None else self.shorcut(x)
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        # out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class ImageEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3, bias=False)  # Adjusted stride
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(Bottleneck(64, 64, stride=1), Bottleneck(64, 64, stride=1))
        self.layer2 = nn.Sequential(Bottleneck(64, 128, stride=2), Bottleneck(128, 128, stride=1))
        self.layer3 = nn.Sequential(Bottleneck(128, 256, stride=2), Bottleneck(256, 256, stride=1))
        self.avgpool = nn.AvgPool2d(11, stride=1)  # Adjusted kernel size
        self.fc = nn.Linear(256, hidden_size)  # Adjusted input channels

    def forward(self, x):
        x = self.conv1(x)  # (64, 84, 84)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (64, 42, 42)
        x = self.layer1(x)  # (64, 42, 42)
        x = self.layer2(x)  # (128, 21, 21)
        x = self.layer3(x)  # (256, 11, 11)
        x = self.avgpool(x)  # (128, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dfc = nn.Linear(hidden_size, 64 * 5 * 5)  # Reduced to 5x5 spatial size
        self.upsample = nn.Upsample(scale_factor=2)
        self.dconv4 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.dconv3 = nn.ConvTranspose2d(64, 64, 2, padding=0)
        self.dconv2 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.dconv1 = nn.ConvTranspose2d(32, 4, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.dfc(x)
        x = x.view(-1, 64, 5, 5)  # Adjusted for 6x6 feature maps
        x = self.upsample(x)  # 10x10
        x = F.relu(self.dconv4(x))  # 10x10
        x = self.upsample(x)  # 20x20
        x = F.relu(self.dconv3(x))  # 21x21
        x = self.upsample(x)  # 42x42
        x = F.relu(self.dconv2(x))  # 42x42
        x = torch.sigmoid(self.dconv1(x))  # 84x84
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder(768)
        self.decoder = ImageDecoder(768)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def count_parameters(model):
    total_params = 0
    print("Component-wise parameter count:\n")

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        # Format the number with thousands separator
        formatted_num = "{:,}".format(num_params)
        print(f"{name: <40}: {formatted_num}")

    # Format the total number with thousands separator
    formatted_total = "{:,}".format(total_params)
    print("\nTotal parameters:", formatted_total)


if __name__ == "__main__":
    ae = Autoencoder()
    print(ae)
    x = torch.rand(1, 4, 84, 84)
    y = ae(x)
    print(y.shape)
    count_parameters(ae)


# y = torch.rand(1, 3, 224, 224)
# x = torch.rand(1, 128)
# x = Variable(x.cuda())
