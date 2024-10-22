from torch import nn, Tensor

import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

class MambaBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, d_state=64, d_conv=4, expand=2):
        super().__init__()
        dim = out_ch
        self.conv1 = nn.Conv2d(in_ch, dim, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )
        self.norm2 = nn.LayerNorm(dim)
        # self.mamba = VSSBlock(dim)
        self.mlp = FFN(dim, dim)
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(dim, dim * 4)),
        #     ("gelu", QuickGLUE()),
        #     ("c_proj", nn.Linear(dim * 4, dim))
        # ]))
        self.layers = 3

    def forward(self, x):
        x = self.conv1(x)
        # print('x',x.shape)
        # B, L, C = x.shape
        batch_size, C, H, W = x.shape
        x_flat = x.reshape(batch_size, C, -1).permute(0, 2, 1)
        # x_flat = x.permute(0, 2, 1)
        # x_norm = self.norm1(x_flat)
        # x_mamba = self.mamba(x_norm)
        # 残差连接
        for _ in range(self.layers):
            residual_output = x_flat + self.mamba(self.norm1(x_flat))
            residual_norm = self.mlp(self.norm2(residual_output))
            x_flat = residual_output + residual_norm

        # out = self.mlp(residual_norm)
        out = residual_norm.permute(0, 2, 1)
        out = out.reshape(batch_size, C, H, W)

        #
        # x = x.permute(0, 2, 3, 1)
        # x = self.mamba(x)
        # out = x.permute(0, 3, 1, 2)
        return out

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

        # self.dwconv = DWConv(hidden_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.dwconv(x.permute(0,3,1,2)).permute(0,2,3,1)
        x = self.act(x)
        x = self.fc2(x)
        return x
