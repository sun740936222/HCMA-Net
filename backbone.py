import torch.nn as nn
from torch.nn import functional as F


# Basic Block
class ResBlock(nn.Module):
    """
    resnet block
    """
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        """
        :param in_ch:
        :param out_ch:
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # extra module:[b, in_ch, h, w]->[b, out_ch, h, w]
        self.extra = nn.Sequential()
        if in_ch != out_ch:
            self.extra = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # shortcut
        out = self.extra(x) + out
        return out
