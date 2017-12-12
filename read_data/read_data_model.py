# -*- coding: UTF-8 -*-
import os
import random
import cv2
import numpy as np
from PIL import Image


# def read_data(file_list):
#     num = len(file_list)
#     n = int(num/4040)
#     for i in range(n):
#         file_list = file_list[4040*i:4040*(i+1)]
#         label = np.zeros(4040)
#         data = np.zeros([4040,512,512,3],np.uint8)
#         j = 0
#         for name, label_min in file_list:
#             label_min__ = '\\' + label_min
#             name__ = '\\' + name
#             folder_path = os.path.join('%s%s' % (filepath, label_min__))
#             file_path = os.path.join('%s%s' % (folder_path, name__))
#             print(file_path)
#             data[j,:,:,:] = cv2.imread(file_path)
#             label[j] = label_min
#             j += 1
#         print(label.shape)
#         if n!=1:
#             np.savez("data00"+str(i)+".npz", data=data, label=label)
#             print("Save the "+str(i)+"file successfully")
#         else:
#             np.savez("valid_data.npz", data=data, label=label)
#             print("Save valid file successfully")
#     return 0


def read_data(file_list, file_name_save):
    num = len(file_list)
    label = np.zeros(num)
    data = np.zeros([num,512,512,3],np.uint8)
    j = 0
    for name, label_min in file_list:
        label_min__ = '\\' + label_min
        name__ = '\\' + name
        folder_path = os.path.join('%s%s' % (filepath, label_min__))
        file_path = os.path.join('%s%s' % (folder_path, name__))
        print(file_path)
        data[j, :, :, :] = cv2.imread(file_path)
        label[j] = label_min
        j += 1
    print(data.shape)
    print(label.shape)
    np.savez(file_name_save, data=data, label=label)
    print("Save file successfully")
    return 0


N_Label = []
# 遍历指定目录，显示目录下的所有文件名
filepath = "F:\\natong\\natong_data_augmentation"
pathDir = os.listdir(filepath)
for allDir_label in pathDir:
    allDir__ = '\\'+allDir_label
    filepath_min = os.path.join('%s%s' % (filepath, allDir__))
    pathdir_min = os.listdir(filepath_min)
    for alldir_name in pathdir_min:
        filename_label = []
        filename_label.append(alldir_name)
        filename_label.append(allDir_label)
        N_Label.append(filename_label)

random.shuffle(N_Label)
# print(N_Label[11])
# print(len(N_Label))
train_data00 = N_Label[:4040]
train_data01 = N_Label[4040:8080]
train_data02 = N_Label[8080:12120]
train_data03 = N_Label[12120:16160]
valid_data = N_Label[16160:]
# print(len(train_data[4040:8080]))
# print(len(valid_data))
read_data(train_data00, "data_000")
read_data(train_data01, "data_001")
read_data(train_data02, "data_002")
read_data(train_data03, "data_003")
read_data(valid_data, "data_004")
