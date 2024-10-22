import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from libtiff import TIFF
import numpy as np
from scipy.io import loadmat
import cv2
import random
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from collections import Counter
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
import os
import time
import datetime as dt
# from GCFNet import Model
# from Resnet18 import ResNet18
# from tqdm import tqdm

print(torch.cuda.is_available())

start_time = dt.datetime.now().strftime('%F %T')
print("程序开始运行时间：" + start_time)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用哪块GPU运行以下程序‘0’代表第一块，‘1’代表第二块

# 网络超参数
EPOCH = 15
BATCH_SIZE = 16
# BATCH_SIZE1 = 1000
LR = 0.001
# Train_Rate = 0.02  # 训练集和测试集按比例拆分,使用列表堆叠
image_index = 6

# label 对应颜色
labelDict = {
    4: 'xian',
    5: 'xian_whole',
    6: 'huhhot',
    7: 'nanjing',
    9: 'beijing',
    10: 'shanghai'
    # 以后再在此处添加即可
}
colordict = {
    'xian': {
        0: [203, 192, 255],
        1: [14, 132, 241],
        2: [255, 255, 0],
        3: [0, 0, 255],
        4: [51, 102, 153],
        5: [0, 255, 0],
        6: [255, 0, 0]
    },
    'xian_whole': {
        0: [255, 255, 0],
        1: [255, 0, 0],
        2: [127, 255, 0],
        3: [33, 145, 237],
        4: [201, 252, 189],
        5: [0, 0, 255],
        6: [58, 58, 139],
        7: [240, 32, 160],
        8: [221, 160, 221],
        9: [140, 230, 240],
        10: [255, 0, 255],
        11: [0, 255, 255],
    },
    'huhhot': {
        0: [255, 255, 0],
        1: [255, 0, 0],
        2: [33, 145, 237],
        3: [0, 255, 0],
        4: [240, 32, 160],
        5: [221, 160, 221],
        6: [140, 230, 240],
        7: [0, 0, 255],
        8: [0, 255, 255],
        9: [127, 255, 0],
        10: [255, 0, 255]
    },
    'nanjing': {
        0: [255, 255, 0],
        1: [255, 0, 0],
        2: [33, 145, 237],
        3: [0, 255, 0],
        4: [140, 230, 240],
        5: [0, 0, 255],
        6: [240, 32, 160],
        7: [0, 255, 255],
        8: [221, 160, 221],
        9: [127, 255, 0],
        10: [255, 0, 255]
    },
    'beijing': {
        0: [255, 255, 0],
        1: [255, 0, 0],
        2: [33, 145, 237],
        3: [201, 252, 189],
        4: [0, 0, 230],
        5: [0, 255, 0],
        6: [240, 32, 160],
        7: [221, 160, 221],
        8: [140, 230, 240],
        9: [0, 255, 255]
    },
    'shanghai': {
        0: [255, 255, 0],
        1: [255, 0, 0],
        2: [33, 145, 237],
        3: [201, 252, 189],
        4: [0, 0, 230],
        5: [0, 255, 0],
        6: [240, 32, 160],
        7: [221, 160, 221],
        8: [140, 230, 240],
        9: [0, 255, 255]
    }
    # 以后再在此处添加即可
}
# 读取图片、标签
#ms4_tif = TIFF.open('Image/ms4.tif', mode='r')
ms4_tif = TIFF.open('/data1/HKZhang/article2/Image/huhhot/ms4.tif', mode='r')
ms4_np = ms4_tif.read_image()
print('原始ms4图的形状：', np.shape(ms4_np))

#pan_tif = TIFF.open('/data1/HKZhang/article2/Image/huhhot/pan.tif', mode='r')
pan_tif = TIFF.open('/data1/HKZhang/article2/Image/huhhot/pan.tif',mode='r')
pan_np = pan_tif.read_image()
print('原始pan图的形状：', np.shape(pan_np))

# mshpan_tif = TIFF.open('./org_data/image1/MSSPAN_fusion.tif', mode='r')
# mshpan_np = mshpan_tif.read_image()
# print('原始MSHpan图的形状;', np.shape(mshpan_np))


# label_mat = loadmat("./org_data/image1/label.mat")
# label_np = label_mat['label']
# label_np = np.transpose(label_mat['label'])
# label_np = np.load('Image/train.npy')
label_np = np.load('/data1/HKZhang/article2/Image/huhhot/train.npy')
print('训练集label数组形状：', np.shape(label_np))
#label_test_np = np.load('Image/test.npy')
label_test_np = np.load('/data1/HKZhang/article2/Image/huhhot/test.npy')
print('测试集label数组形状：', np.shape(label_test_np))

# ground_truth = cv2.imread('./data/groundtruth.bmp')
# 上色图定义
out_color = np.zeros((np.shape(ms4_np)[0], np.shape(ms4_np)[1], 3))
out_color_bufen = np.zeros(
    (np.shape(ms4_np)[0], np.shape(ms4_np)[1], 3))  # 高度 * 宽度 * 3


# ms4与pan图补零
Ms4_patch_size = 16  # ms4截块的边长
Interpolation = cv2.BORDER_REFLECT_101
# cv2.BORDER_REPLICATE： 进行复制的补零操作;
# cv2.BORDER_REFLECT:  进行翻转的补零操作:gfedcba|abcdefgh|hgfedcb;
# cv2.BORDER_REFLECT_101： 进行翻转的补零操作:gfedcb|abcdefgh|gfedcb;
# cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作:bcdegh|abcdefgh|abcdefg;

top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的ms4图的形状：', np.shape(ms4_np))

Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的pan图的形状：', np.shape(pan_np))

# MSHPan_patch_size = Ms4_patch_size * 4  # MSHpan截块的边长
# top_size, bottom_size, left_size, right_size = (int(MSHPan_patch_size/2-4), int(MSHPan_patch_size/2),
#                                                int(MSHPan_patch_size/2-4), int(MSHPan_patch_size/2))
# mshpan_np = cv2.copyMakeBorder(mshpan_np, top_size, bottom_size, left_size, right_size, Interpolation)
# print('补零后的MSHpan图的形状：', np.shape(mshpan_np))

# 按类别比例拆分数据集
# label_np=label_np.astype(np.uint8)
label_np = label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255
label_test_np = label_test_np - 1

label_element, element_count = np.unique(label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
label_test_element, element_test_count = np.unique(label_test_np, return_counts=True)  # 返回测试集类别标签与各个类别所占的数量
print('类标：', label_element)
print('训练集各类样本数：', element_count)
Categories_Number = len(label_element) - 1  # 数据的类别数
print('训练集标注的类别数：', Categories_Number)

print('测试集类标：', label_test_element)
print('测试集各类样本数：', element_test_count)
Categories_Number_test = len(label_test_element) - 1  # 数据的类别数
print('测试集标注的类别数：', Categories_Number_test)

label_row, label_column = np.shape(label_np)  # 获取标签图的行、列
label_test_row, label_test_column = np.shape(label_test_np)  # 获取标签图的行、列
'''归一化图片'''
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image
ground_xy = np.array([[]] * Categories_Number).tolist()
ground_test_xy = np.array([[]] * Categories_Number_test).tolist()
ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)

count = 0
for row in range(label_row):  # 行
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np[row][column] != 255:
            ground_xy[int(label_np[row][column])].append([row, column])
count1 = 0
for row in range(label_test_row):  # 行
    for column in range(label_test_column):
        ground_xy_allData[count1] = [row, column]
        count1 = count1 + 1
        if label_test_np[row][column] != 255:
            ground_test_xy[int(label_test_np[row][column])].append([row, column])

# 标签内打乱
for categories in range(Categories_Number):
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
    np.random.shuffle(shuffle_array)

    ground_xy[categories] = ground_xy[categories][shuffle_array]
shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]
for categories in range(Categories_Number_test):
    ground_test_xy[categories] = np.array(ground_test_xy[categories])
    shuffle_test_array = np.arange(0, len(ground_test_xy[categories]), 1)
    np.random.shuffle(shuffle_test_array)

    ground_test_xy[categories] = ground_test_xy[categories][shuffle_test_array]
shuffle_test_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_test_array)

ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    # print('aaa', categories_number)
    for i in range(categories_number):
        if i < int(categories_number):
            ground_xy_train.append(ground_xy[categories][i])
    label_train = label_train + [categories for x in range(int(categories_number))]
for categories in range(Categories_Number_test):
    categories_test_number = len(ground_test_xy[categories])
    # print('aaa', categories_number)
    for i in range(categories_test_number):
        if i < int(categories_test_number):
            ground_xy_test.append(ground_test_xy[categories][i])
    label_test = label_test + [categories for x in range(int(categories_test_number))]

label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_train = np.array(ground_xy_train)
ground_xy_test = np.array(ground_xy_test)

# 训练数据与测试数据，数据集内打乱
shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]

label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

print('训练样本数：', len(label_train))
print('测试样本数：', len(label_test))

# 数据归一化
ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
# mshpan = to_tensor(mshpan_np)
pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
# mshpan = np.expand_dims(mshpan, axis=0)# 二维数据进网络前要加一维
ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道

# 转换类型
ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)


class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        # self.train_data3 = MSHPAN
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size*4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4*x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4*y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        # image_mshpan = self.train_data3[:, x_pan:x_pan+self.cut_pan_size,
        #                 y_pan:y_pan+self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]
        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)


class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        # self.train_data3 = MSHPAN
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size*4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4*x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4*y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        # image_mshpan = self.train_data3[:, x_pan:x_pan+self.cut_pan_size,
        #                  y_pan:y_pan+self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)


# cnn = ResNet18()
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # LR = LR * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.9995

#
# # 参数初始化方法
# for m in Model.modules(cnn):
#    if isinstance(m, (nn.Conv2d,nn.Linear)):
#        nn.init.xavier_uniform_(m.weight)
#
# cnn.cuda()


# # # loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
loss_func = nn.CrossEntropyLoss()
train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)
#
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_data_loader = Data.DataLoader(dataset=all_data, batch_size=BATCH_SIZE * 5, shuffle=False, num_workers=0)

class_count = np.zeros(Categories_Number)

# 生成着色图
transformer = torch.load('/data1/HKZhang/article2/Result/huhhot_1_2_3_75.26_layer_3/huhhot_best_7_baseline-IRL-Net.pkl')
transformer.cuda()

# print('begin generare out_quanse..............')
# for step, (ms, pan, gt_xy) in enumerate(all_data_loader):
#     ms = ms.cuda()
#     pan = pan.cuda()
#
#     with torch.no_grad():
#         output = transformer(ms, pan, 'test')  # cnn output
#
#     pred_y = torch.argmax(output[0], 1)
#     pred_y_numpy = pred_y.cpu().numpy()
#     gt_xy = gt_xy.numpy()
#     for k in range(len(gt_xy)):
#         class_count[pred_y_numpy[k]] = class_count[pred_y_numpy[k]] + 1
#         out_color[gt_xy[k][0]][gt_xy[k][1]] = colordict[
#             labelDict[image_index]][pred_y_numpy[k]]
#
# cv2.imwrite("out_quanse.png", out_color)


# 生成部分着色图
print('begin generare out_bufen..............')
class_count = np.zeros(Categories_Number)
for step, (ms, pan, label, gt_xy) in enumerate(test_loader):
    ms = ms.cuda()
    pan = pan.cuda()
    label = label.cuda()
    with torch.no_grad():
        output = transformer(ms, pan, 'test')  # cnn output

    pred_y = torch.argmax(output[0], 1)
    pred_y_numpy = pred_y.cpu().numpy()
    gt_xy = gt_xy.numpy()
    for k in range(len(gt_xy)):
        class_count[pred_y_numpy[k]] = class_count[pred_y_numpy[k]] + 1
        # label_test = label_test + [categories for x in range(categories_number-int(categories_number*Train_Rate))]
        out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = colordict[
            labelDict[image_index]][pred_y_numpy[k]]

for step, (ms, pan, label, gt_xy) in enumerate(train_loader):
    ms = ms.cuda()
    pan = pan.cuda()
    label = label.cuda()
    with torch.no_grad():
        output = transformer(ms, pan, 'test')  # cnn output

    pred_y = torch.argmax(output[0], 1)
    pred_y_numpy = pred_y.cpu().numpy()
    gt_xy = gt_xy.numpy()
    for k in range(len(gt_xy)):
        class_count[pred_y_numpy[k]] = class_count[pred_y_numpy[k]] + 1
        out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = colordict[
            labelDict[image_index]][pred_y_numpy[k]]

cv2.imwrite("out_bufen_hohhot.png", out_color_bufen)

end_time = dt.datetime.now().strftime('%F %T')
print("end time:" + end_time)
