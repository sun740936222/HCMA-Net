import torch
import torch.nn as nn
import torch.nn.functional as F

from CRSL import Complementary_Learning_Loss
from backbone import ResBlock
from resnet import resnet18
from mambablock import MambaBlock
from einops import rearrange
from timm.models.vision_transformer import Block
from crosstransformer import CrossTransformer
from torchinfo import summary
class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet18(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class CN_Layer(nn.Module):
    """Cross-Nonlocal Layer, CNL"""
    """交叉非局部层，用于捕捉不同特征空间中的长距离依赖关系。"""
    def __init__(self, high_dim, low_dim, flag=0):
        super(CN_Layer, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        # 定义g卷积层，用于生成注意力机制中的g特征
        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        # 定义theta卷积层，用于生成注意力机制中的theta特征
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        # 根据flag的不同，定义phi卷积层和W序列
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(high_dim), )
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                                   nn.BatchNorm2d(self.high_dim), )
        # 初始化批量归一化层的权重和偏置
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        # 通过g卷积层生成g特征并转换为矩阵形式
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        # 通过theta卷积层生成theta特征，通过phi卷积层生成phi特征，并转换为矩阵形
        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        # 注意phi_x需要转置以匹配矩阵乘法的维度要求
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        # 计算能量矩阵，即theta和phi特征的矩阵乘法结果
        energy = torch.matmul(theta_x, phi_x)
        # 归一化能量矩阵以获得注意力分数
        attention = energy / energy.size(-1)

        # 通过注意力分数和g特征进行矩阵乘法，得到加权的g特征
        y = torch.matmul(attention, g_x)
        # 将结果重新转换为张量形式，以匹配原始输入的维度
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        # 通过W序列对y进行变换，以提升特征维度并进行批量归一化
        W_y = self.W(y)
        # 将变换后的特征与原始高维特征相加，实现特征融合
        z = W_y + x_h

        return z


class PN_Layer(nn.Module):
    """Pixel Nonlocal Layer,PNL"""
    """像素非局部层，用于捕捉空间上的长距离依赖关系。"""
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PN_Layer, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        # 定义g, theta, phi卷积层，用于生成注意力机制中的特征
        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim), )
        # 初始化批量归一化层的权重和偏置
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)

        # 通过g, theta, phi卷积层生成对应的特征并转换为矩阵形式
        g_x = self.g(x_l).reshape(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).reshape(B, self.low_dim, -1)

        # 计算能量矩阵，即theta和phi特征的矩阵乘法结果
        energy = torch.matmul(theta_x, phi_x)
        # 归一化能量矩阵以获得注意力分数
        attention = energy / energy.size(-1)

        #通过注意力分数和g特征进行矩阵乘法，得到加权的g特征
        y = torch.matmul(attention, g_x)
        # 将结果重新转换为张量形式，以匹配原始输入的维度
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        # 通过W序列对y进行变换，以提升特征维度并进行批量归一化
        W_y = self.W(y)
        # 将变换后的特征与原始高维特征相加，实现特征融合
        z = W_y + x_h
        return z



class MFI_block(nn.Module):
    "Multiphase Feature Integration Block (MFIB)"
    def __init__(self, high_dim, low_dim, flag):
        super(MFI_block, self).__init__()

        self.CN_L = CN_Layer(high_dim, low_dim, flag)  # 实例化交叉非局部层
        self.PN_L = PN_Layer(high_dim, low_dim)  # 实例化像素非局部层

    def forward(self, x, x0):
        # 通过交叉非局部层
        z = self.CN_L(x, x0)
        # 再通过像素非局部层
        z = self.PN_L(z, x0)
        return z


class MIFE(nn.Module):
    "Modular Interaction Feature Extractor (MIFE)"
    def __init__(self):
        super(MIFE, self).__init__()
        self.FC = nn.Linear(512 * 2, 512)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + (2 * 16 * 16), 512))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.SA_T = nn.ModuleList([
            Block(512, 4, 4.0, qkv_bias=False, norm_layer=nn.LayerNorm)
            for i in range(4)])
        self.FC2 = nn.Linear(512, 512)
        # Decoder
        self.CA_T = CrossTransformer()
        self.FC3 = nn.Linear(512, 512)

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        # 调整特征形状以匹配期望的输入格式
        feat1 = rearrange(feat1, 'B C H W -> B (H W) C')
        feat2 = rearrange(feat2, 'B C H W -> B (H W) C')

        # PAN-MS Fusion Feature
        # 使用全连接层合并PAN和MS特征
        fusion_feat = self.FC(torch.cat((feat1, feat2), dim=-1))

        # PAN-MS Interaction Feature
        # 将PAN和MS特征拼接，并加上位置嵌入
        interaction_input = torch.cat((feat1, feat2), dim=1) + self.pos_embed[:, 1:]
        cls_token = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        interaction_feat = torch.cat((cls_token, interaction_input), dim=1)

        # 通过自注意力模块处理交互特征
        for blk in self.SA_T:
            interaction_feat = blk(interaction_feat)
        interaction_feat = self.FC2(interaction_feat)

        # 使用交叉注意力Transformer处理融合特征和交互特征
        output = self.CA_T(fusion_feat, interaction_feat)
        # 通过一个全连接层进一步处理输出特征
        output = self.FC3(output)
        # 调整输出形状以恢复到原始的B C H W格式
        output = rearrange(output, 'B (H W) C -> B C H W', H=H, W=W)
        return output

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=11,arch='resnet50'):
        super(ResNet, self).__init__()

        self.in_planes = 64

        # self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=64, kernel_size=4, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)


        self.bn2 = nn.BatchNorm2d(64)

        self.layer1_1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_1 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3_1 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4_1 = self._make_layer(block, 512, num_blocks[3], stride=1)

        self.in_planes = 64
        self.layer1_2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3_2 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4_2 = self._make_layer(block, 512, num_blocks[3], stride=1)

        self.base_resnet = base_resnet(arch=arch)
        self.linear = nn.Linear(1024, num_classes)

        # 初始化多阶段特征整合块，用于整合不同层级的特征
        self.MFI1 = MFI_block(64, 64, 0)
        self.MFI2 = MFI_block(128, 64, 1)
        self.MFI3 = MFI_block(256, 128, 1)
        self.MFI4 = MFI_block(512, 256, 0)

        # 初始化Mamba块，用于特征的进一步提取和整合
        self.mb1 = MambaBlock(64, 64)
        self.mb2 = MambaBlock(64, 128, stride=2)
        self.mb3 = MambaBlock(128, 256, stride=2)
        self.mb4 = MambaBlock(512, 512)

        # 引入MIFE模块，用于提取多模态交互特征
        self.MIFE = MIFE()
        # 定义最终的特征融合卷积层
        self.conv_ms = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_pan = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def feature_loss(self, x, y):
        # 定义特征损失函数，用于训练过程中的特征比较
        loss = Complementary_Learning_Loss(x, y).cuda()
        loss_mean = torch.mean(loss)
        loss_std = torch.std(loss)
        loss_new = (loss - loss_mean) / loss_std
        loss_new = torch.mean(loss_new)
        return loss_new

    def forward(self, x, y, phase):
        # x 和 y 分别为网络的两个输入，例如多光谱图像和全色图像
        # phase 表示当前的阶段，'train' 表示训练阶段，其他值可能表示测试或其他阶段

        # 对第一个输入 x 进行转置卷积、批量归一化和ReLU激活操作
        x_1 = F.relu(self.bn1(self.conv1(x)))
        # 对第二个输入 y 进行卷积、批量归一化和ReLU激活操作
        y_1 = F.relu(self.bn2(self.conv2(y)))
        # 将两个输入的特征图沿批次维度合并
        f_x = torch.cat([x_1, y_1], dim=0)

        f_m0 = f_x
        # 通过基础的ResNet模型提取特征提取第一层特征，并将Mamba块的输出与ResNet层的输出相加
        f_x_1_1 = self.base_resnet.base.layer1(f_m0)
        f_x_1_2 = self.mb1(f_m0)

        f_x_1_2 = f_x_1_1 + f_x_1_2

        # 将特征输入到多阶段特征整合块 MFI1，并获取整合后的特征
        f_m_x_1 = self.MFI1(f_m0, f_x_1_2)

        f_x_1, f_y_1 = f_m_x_1.chunk(2, dim=0)

        f_x_2_1 = self.base_resnet.base.layer2(f_m_x_1)
        f_x_2_2 = self.mb2(f_m_x_1)

        f_x_2_2 = f_x_2_1 + f_x_2_2
        f_mx_2_2 = self.MFI2(f_x_2_2, f_m_x_1)
        f_m_x_2 = f_mx_2_2
        f_x_2, f_y_2 = f_m_x_2.chunk(2, dim=0)

        f_x_3_1 = self.base_resnet.base.layer3(f_m_x_2)
        f_x_3_2 = self.mb3(f_m_x_2)

        f_x_3_2 = f_x_3_1 + f_x_3_2
        f_mx_3_2 = self.MFI3(f_x_3_2, f_m_x_2)
        f_m_x_3 = f_mx_3_2
        f_x_3, f_y_3 = f_m_x_3.chunk(2, dim=0)

        f_m_x_4 = self.base_resnet.base.layer4(f_m_x_3)
        # 将整合后的特征沿批次维度分为两部分，分别对应原始的两个输入
        f_x_4, f_y_4 = f_m_x_4.chunk(2, dim=0)

        inter_feat = self.MIFE(f_x_4, f_y_4)

        f_x_4 = f_x_4 + inter_feat
        f_y_4 = f_y_4 + inter_feat

        out = []
        if phase == 'train':
            loss = self.feature_loss(f_x_4, f_y_4)
            out.append(loss)
        # if phase == 'train':
        #     loss1 = self.feature_loss(f_x_1, f_x_1)
        #     loss2 = self.feature_loss(f_x_2, f_x_2)
        #     loss3 = self.feature_loss(f_x_3, f_x_3)
        #     loss4 = self.feature_loss(f_x_4, f_y_4)
        #     loss = torch.mean(loss1) + torch.mean(loss2) + torch.mean(loss3) + torch.mean(loss4)
        #     out.append(loss)

        # 对最终的特征图进行全局平均池化，将特征图的大小降为 [1, 1]
        f_x_5 = F.adaptive_avg_pool2d(f_x_4, [1, 1])
        f_y_5 = F.adaptive_avg_pool2d(f_y_4, [1, 1])

        # 将两个输入的全局平均池化后的特征沿特征通道维度合并
        rel = torch.cat([f_x_5, f_y_5], dim=1)
        # 将合并后的特征展平，并传递给分类器
        rel = rel.view(rel.size(0), -1)
        rel = self.linear(rel)
        out.append(rel)
        return out


def ResNet18(num_classes=12):
    return ResNet(ResBlock, [2, 2, 2, 2],num_classes=num_classes)
