# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
from matplotlib import pyplot


# 遍历指定目录，显示目录下的所有文件名
filepath = "E:\\lenovo_exercitation\\natong_work\\natong_product\\natong_product"
pathDir = os.listdir(filepath)
for allDir in pathDir:
    allDir__ = '\\'+allDir
    filepath_min = os.path.join('%s%s' % (filepath, allDir__))
    filepath_save = os.path.join('%s%s' % ('E:\\lenovo_exercitation\\natong_work\\natong_product\\natong_data_augmentation\\natong_data_save_resize', allDir__))
    # filepath_save = filepath_save + '_mf'
    # print (filepath_save)            # child.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题
    pathdir_min = os.listdir(filepath_min)
    # # 保存路径
    # isExists = os.path.exists(filepath_save)
    # # 判断结果
    # if not isExists:
    #     # 如果不存在则创建目录
    #     os.makedirs(filepath_save)

    for alldir_min in pathdir_min:
        alldir_min__ = '\\' + alldir_min

        filepath_final = os.path.join('%s%s' % (filepath_min, alldir_min__))

        # print(filepath_final)
        read_path = filepath_final
        img = cv2.imread(read_path,1)
        img = cv2.resize(img, (960, 540))



        # #保存
        filepath_min_save = filepath_save + "\\"+alldir_min
        print(filepath_min_save)
        cv2.imwrite(filepath_min_save,img, params=None)















# import Image
#
# infile = 'D:\\original_img.jpg'
# outfile = 'D:\\adjust_img.jpg'
# im = Image.open(infile)
# (x, y) = im.size  # read image size
# x_s = 250  # define standard width
# y_s = y * x_s / x  # calc height based on standard width
# out = im.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
# out.save(outfile)

# print
# 'original size: ', x, y
# print
# 'adjust size: ', x_s, y_s