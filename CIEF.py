import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from CRSL import Complementary_Learning_Loss, Cos_Similarity


def Downsample(x, y):
    _, _, h1, w1 = x.size()
    downsample = nn.AdaptiveAvgPool2d(output_size=(h1, w1))
    result = downsample(y)
    return result


def L2Norm(x):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(x, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(x)
    return torch.div(x, norm)


def MutualMatching(corr4d):
    # get max
    corr4d_B_max, _ = torch.max(corr4d, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d, dim=2, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d / (corr4d_B_max + eps)
    corr4d_A = corr4d / (corr4d_A_max + eps)

    spacial_corr4d = corr4d * (corr4d_A * corr4d_B)

    return spacial_corr4d


class Cross_Modal_Interaction(nn.Module):
    def __init__(self, in_planes):
        super(Cross_Modal_Interaction, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(in_planes)

        self.conv4 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(in_planes)
        self.conv5 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(in_planes)
        self.conv6 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1)
        self.bn6 = nn.BatchNorm2d(in_planes)

        self.extra = nn.Sequential()
        self.softmax = nn.Softmax(dim=-1)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.Sigmoid()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.Sigmoid()
        )

    def forward(self, ms, pan):
        b1, c1, h1, w1 = pan.size()
        b2, c2, h2, w2 = ms.size()

        pan_1 = F.relu(self.bn1(self.conv1(pan)))
        pan_1 = self.extra(pan) + pan_1
        pan_k = pan_1.view(b1, c1, h1 * w1)

        pan_2 = F.relu(self.bn2(self.conv2(pan)))
        pan_2 = self.extra(pan) + pan_2
        pan_q = pan_2.view(b1, c1, h1 * w1).transpose(1, 2)

        pan_3 = F.relu(self.bn3(self.conv3(pan)))
        pan_3 = self.extra(pan) + pan_3
        pan_v = pan_3.view(b1, c1, h1 * w1)

        ms_1 = F.relu(self.bn4(self.conv4(ms)))
        ms_1 = self.extra(ms) + ms_1
        ms_k = ms_1.view(b2, c2, h2 * w2)

        ms_2 = F.relu(self.bn5(self.conv5(ms)))
        ms_2 = self.extra(ms) + ms_2
        ms_q = ms_2.view(b2, c2, h2 * w2).transpose(1, 2)

        ms_3 = F.relu(self.bn6(self.conv6(ms)))
        ms_3 = self.extra(ms) + ms_3
        ms_v = ms_3.view(b2, c2, h2 * w2)

        # Prejust
        pan_attn1 = L2Norm(torch.matmul(pan_q, pan_k) / math.sqrt(pan_q.size(-1)))
        ms_attn1 = L2Norm(torch.matmul(ms_q, ms_k) / math.sqrt(ms_q.size(-1)))

        # Intra-modal Strenghthen
        pan_attn = self.extra(pan_attn1) + MutualMatching(pan_attn1)
        ms_attn = self.extra(ms_attn1) + MutualMatching(ms_attn1)

        # Inter-modal Enhanced Fusion
        attn = self.softmax(torch.mul(pan_attn, ms_attn))

        ms_to_pan = torch.matmul(ms_v, attn).view(b1, c1, h1, w1)
        pan_to_ms = torch.matmul(pan_v, attn).view(b2, c2, h2, w2)

        # Update
        ms_to_pan_1 = self.block1(ms_to_pan)
        pan_to_ms_1 = self.block2(pan_to_ms)

        ms_to_pan_2 = torch.mul(ms_to_pan_1, ms_to_pan)
        pan_to_ms_2 = torch.mul(pan_to_ms_1, pan_to_ms)

        ms_out = torch.add(ms, ms_to_pan_2)
        pan_out = torch.add(pan, pan_to_ms_2)

        # Complementary_Learning_Loss
        loss = Complementary_Learning_Loss(ms_3, pan_3).cuda()
        loss_mean = torch.mean(loss)
        loss_std = torch.std(loss)
        loss_new = (loss - loss_mean) / loss_std

        return ms_out, pan_out, loss_new

