# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
from matplotlib import pyplot
from skimage import filters

def salt(img, n):
    for k in range(n):
        i = int(np.random.random() * img.shape[1]);
        j = int(np.random.random() * img.shape[0]);
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
    return img

# 遍历指定目录，显示目录下的所有文件名
filepath = "F:\\baidudownload\\natong__20171115\\natong_image\\natong"
pathDir = os.listdir(filepath)
for allDir in pathDir:
    allDir__ = '\\'+allDir
    filepath_min = os.path.join('%s%s' % (filepath, allDir__))
    filepath_save = os.path.join('%s%s' % ('E:\\lenovo_exercitation\\natong_work\\natong_product\\natong_product', allDir__))
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

        img = cv2.imread(filepath_final)
        mask = np.zeros(img.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
    #制作mask
        mask_old = cv2.imread(filepath_final, 0)
        #中值滤波
        img02 = salt(mask_old, 500)
        mask_old = cv2.medianBlur(img02, 5)
        #阈值化
        thresh = filters.threshold_li(mask_old)
        newmask = (mask_old <= thresh) * 1.0  # 阈值化

        # whereever it is marked white (sure foreground), change mask=1
        # whereever it is marked black (sure background), change mask=0
        mask[newmask == 0] = 1
        mask[newmask == 1] = 0
        mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img, img, mask=mask2)
        #中值滤波
        output = salt(output, 500)
        output = cv2.medianBlur(output, 5)
        # #保存
        filepath_min_save = filepath_save + "\\"+alldir_min
        print(filepath_min_save)
        cv2.imwrite(filepath_min_save,output, params=None)
