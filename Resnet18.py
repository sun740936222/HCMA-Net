import torch
import torch.nn as nn
from CIEF import *
from backbone import ResBlock


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=11):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.layer1_1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_1 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3_1 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4_1 = self._make_layer(block, 512, num_blocks[3], stride=1)

        self.in_planes = 64
        self.layer1_2 = self._make_layer(block, 64, num_blocks[4], stride=1)
        self.layer2_2 = self._make_layer(block, 128, num_blocks[5], stride=1)
        self.layer3_2 = self._make_layer(block, 256, num_blocks[6], stride=1)
        self.layer4_2 = self._make_layer(block, 512, num_blocks[7], stride=1)

        self.interaction1 = Cross_Modal_Interaction(64).cuda()
        self.interaction2 = Cross_Modal_Interaction(128).cuda()
        self.interaction3 = Cross_Modal_Interaction(256).cuda()
        self.interaction4 = Cross_Modal_Interaction(512).cuda()

        self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y, phase):
        x_1 = F.relu(self.bn1(self.conv1(x)))
        y_1 = F.relu(self.bn2(self.conv2(y)))

        # 单模态特征提取
        f_x_1 = self.layer1_1(x_1)
        f_y_1 = self.layer1_2(y_1)
        f_y_1 = Downsample(f_x_1, f_y_1)
        f_x_1, f_y_1, loss1 = self.interaction1(f_x_1, f_y_1)

        f_x_2 = self.layer2_1(f_x_1)
        f_y_2 = self.layer2_2(f_y_1)
        f_y_2 = Downsample(f_x_2, f_y_2)
        f_x_2, f_y_2, loss2 = self.interaction2(f_x_2, f_y_2)

        f_x_3 = self.layer3_1(f_x_2)
        f_y_3 = self.layer3_2(f_y_2)
        f_y_3 = Downsample(f_x_3, f_y_3)
        f_x_3, f_y_3, loss3 = self.interaction3(f_x_3, f_y_3)

        f_x_4 = self.layer4_1(f_x_3)
        f_y_4 = self.layer4_2(f_y_3)
        f_y_4 = Downsample(f_x_4, f_y_4)
        f_x_4, f_y_4, loss4 = self.interaction4(f_x_4, f_y_4)

        out = []
        if phase == 'train':
            loss = torch.mean(loss1) + torch.mean(loss2) + torch.mean(loss3) + torch.mean(loss4)
            out.append(loss)

        f_x_5 = F.adaptive_avg_pool2d(f_x_4, [1, 1])
        f_y_5 = F.adaptive_avg_pool2d(f_y_4, [1, 1])
        rel = torch.cat([f_x_5, f_y_5], dim=1)
        rel = rel.view(rel.size(0), -1)
        rel = self.linear(rel)
        out.append(rel)
        return out


def ResNet18(num_classes=12):
    return ResNet(ResBlock, [2, 2, 2, 2, 2, 2, 2, 2],num_classes=num_classes)