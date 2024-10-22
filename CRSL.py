import torch
import torch.nn.functional as F
from torch import nn


def Norm(x):
    max_val_t = torch.max(x, 2)[0]
    max_val = torch.max(max_val_t, 2)[0]

    min_val_t = torch.min(x, 2)[0]
    min_val = torch.min(min_val_t, 2)[0]

    delta_t1 = torch.sub(max_val, min_val)
    delta_t2 = torch.unsqueeze(delta_t1, 2)
    delta = torch.unsqueeze(delta_t2, 3)

    min_val_t1 = torch.unsqueeze(min_val, 2)
    min_val = torch.unsqueeze(min_val_t1, 3)

    rel_t1 = torch.sub(x, min_val)
    rel_t2 = torch.div(rel_t1, delta)
    rel = torch.mul(rel_t2, 255).int()
    return rel


def Entropy(x):
    B, C, W, H = x.size()
    size = W * H
    histic = torch.zeros(size=(B, C, 256))
    for i in range(256):
        eq_i = torch.eq(x, i)
        sum_t1 = torch.sum(eq_i, dim=2)
        sum = torch.sum(sum_t1, dim=2)
        histic[:, :, i] = sum
    p_ij = torch.div(histic, size)
    h_ij_t1 = torch.add(p_ij, 1e-8)
    h_ij_t2 = p_ij * torch.log(h_ij_t1)
    h_ij = -torch.sum(h_ij_t2, dim=2)
    return torch.unsqueeze(torch.unsqueeze(h_ij, 2), 3)


def Cos_Similarity(x, y):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(x, y)


def Smooth(tensor, a=0.02):
    temp_norm = torch.norm(tensor.type(torch.FloatTensor), p=2, dim=1)
    map_1 = (temp_norm >= a).float()
    map_s = (temp_norm < a).float()
    comb = map_1 * temp_norm + map_s * (0.5 * torch.pow(temp_norm, 2)/a + a * 0.5)
    return comb


def Complementary_Learning_Loss(ms_v, pan_v):
    ms_v = torch.mean(ms_v, dim=1, keepdim=True)
    pan_v = torch.mean(pan_v, dim=1, keepdim=True)

    # 特征的余弦相似度
    cos_similarity = Cos_Similarity(ms_v, pan_v).cuda()

    # 特征差异性
    ms_v = Norm(ms_v).cuda()
    pan_v = Norm(pan_v).cuda()
    ms_v_imp = Entropy(ms_v).cuda()
    pan_v_imp = Entropy(pan_v).cuda()

    x = torch.sub(ms_v_imp, pan_v_imp)
    k = torch.ones_like(cos_similarity) / (torch.pow(cos_similarity, 2) + 0.001)

    # 惩罚权重
    w1 = torch.sigmoid(k * x)
    w2 = 1 - w1

    loss = w1 * Smooth(ms_v).cuda() + w2 * Smooth(pan_v).cuda()

    return loss
