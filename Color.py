import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
from libtiff import TIFF
import numpy as np
import cv2
import os
import datetime as dt

start_time = dt.datetime.now().strftime('%F %T')
print("begin time:" + start_time)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

EPOCH = 30
BATCH_SIZE = 48
LR = 0.001
Train_Rate = 0.02

# 图片目录索引
image_index = 6

# label 对应颜色
labelDict = {
    6: 'hohhot',
}
colordict = {
    'hohhot': {
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
    }
}
# 读取图片、标签
ms4_tif = TIFF.open('./Image/ms4.tif', mode='r')
ms4_np = ms4_tif.read_image()
print('orginal ms4 shape:', np.shape(ms4_np))

pan_tif = TIFF.open('./Image/pan.tif', mode='r')
pan_np = pan_tif.read_image()
print('orginal pan shape:', np.shape(pan_np))

label_np = np.load("./Image/label6.npy")
print('label shape:', np.shape(label_np))

# 上色图定义
out_color = np.zeros((np.shape(ms4_np)[0], np.shape(ms4_np)[1], 3))
out_color_bufen = np.zeros(
    (np.shape(ms4_np)[0], np.shape(ms4_np)[1], 3))  # 高度 * 宽度 * 3

# ms4与pan图补零
Ms4_patch_size = 16
Interpolation = cv2.BORDER_REFLECT_101

top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1),
                                                int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1),
                                                int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size,
                            right_size, Interpolation)
print('padding ms4 shape:', np.shape(ms4_np))

Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4),
                                                int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 4),
                                                int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size,
                            right_size, Interpolation)
print('padding pan shape:', np.shape(pan_np))

# 按类别比例拆分数据集
# label_np=label_np.astype(np.uint8)
label_np = label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255

label_element, element_count = np.unique(
    label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
print('label:', label_element)
print('label count:', element_count)
Categories_Number = len(label_element) - 1  # 数据的类别数
print('label classes:', Categories_Number)
label_row, label_column = np.shape(label_np)  # 获取标签图的行、列
'''归一化图片'''


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


ground_xy = np.array([[]] * Categories_Number).tolist()
ground_xy_allData = np.arange(label_row * label_column * 2).reshape(
    label_row * label_column, 2)

count = 0
for row in range(label_row):  # 行
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np[row][column] != 255:
            ground_xy[int(label_np[row][column])].append([row, column])

# 标签内打乱
for categories in range(Categories_Number):
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
    np.random.shuffle(shuffle_array)

    ground_xy[categories] = ground_xy[categories][shuffle_array]
shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]

ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    # print('aaa', categories_number)
    for i in range(categories_number):
        if i < int(categories_number * Train_Rate):
            ground_xy_train.append(ground_xy[categories][i])
        else:
            ground_xy_test.append(ground_xy[categories][i])
    label_train = label_train + [
        categories for x in range(int(categories_number * Train_Rate))
    ]
    label_test = label_test + [
        categories
        for x in range(categories_number - int(categories_number * Train_Rate))
    ]

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

print('train sample count:', len(label_train))
print('test sample count', len(label_test))

# 数据归一化
ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)

pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维

ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道

# 转换类型
ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)


class MyData(Dataset):

    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]
        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)


class MyData1(Dataset):

    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)


train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=BATCH_SIZE * 4,
                               shuffle=False,
                               num_workers=0)
test_loader = Data.DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE * 4,
                              shuffle=False,
                              num_workers=0)
all_data_loader = Data.DataLoader(dataset=all_data,
                                  batch_size=BATCH_SIZE * 4,
                                  shuffle=False,
                                  num_workers=0)

class_count = np.zeros(Categories_Number)

# 生成着色图
transformer = torch.load('./Models/I³RL-Net.pkl')
transformer.cuda()

print('begin generare out_quanse..............')
for step, (ms, pan, gt_xy) in enumerate(all_data_loader):
    ms = ms.cuda()
    pan = pan.cuda()

    with torch.no_grad():
        output = transformer(ms, pan, 'test')  # cnn output

    pred_y = torch.argmax(output[0], 1)
    pred_y_numpy = pred_y.cpu().numpy()
    gt_xy = gt_xy.numpy()
    for k in range(len(gt_xy)):
        class_count[pred_y_numpy[k]] = class_count[pred_y_numpy[k]] + 1
        out_color[gt_xy[k][0]][gt_xy[k][1]] = colordict[
            labelDict[image_index]][pred_y_numpy[k]]

cv2.imwrite("out_quanse.png", out_color)


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

cv2.imwrite("out_bufen.png", out_color_bufen)

end_time = dt.datetime.now().strftime('%F %T')
print("end time:" + end_time)
