import torch.nn as nn


class FeatureModule(nn.Module):
    def __init__(self):
        super(FeatureModule, self).__init__()
        self.conv1 = nn.Sequential([
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Relu(inplace=True),
        ])
        self.conv2 = nn.Sequential([
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Relu(inplace=True),
        ])
        self.conv3 = nn.Sequential([
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Relu(inplace=True),
        ])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return self.maxpool(x)
