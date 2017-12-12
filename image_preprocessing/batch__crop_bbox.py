# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
from PIL import Image


# 遍历指定目录，显示目录下的所有文件名
filepath = "F:\\natong\\grabcut_grabcut_skew"
pathDir = os.listdir(filepath)
for allDir in pathDir:
    allDir__ = '\\'+allDir
    filepath_min = os.path.join('%s%s' % (filepath, allDir__))
    filepath_save = os.path.join('%s%s' % ('F:\\natong\\natong_product_skew_crop', allDir__))
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

        image = cv2.imread(filepath_final)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur and threshold the image
        blurred = cv2.blur(gray, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
        # 形态学方面的操作填充内部白色区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        # box = np.int0(cv2.cv.BoxPoints(rect))
        # draw a bounding box arounded the detected barcode and display the image
        # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        cropImg = image[y1:y1 + hight, x1:x1 + width]
        #保存
        filepath_min_save = filepath_save + "\\"+alldir_min
        print(filepath_min_save)
        cv2.imwrite(filepath_min_save, cropImg)# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
from PIL import Image


# 遍历指定目录，显示目录下的所有文件名
filepath = "F:\\natong\\grabcut_grabcut_skew"
pathDir = os.listdir(filepath)
for allDir in pathDir:
    allDir__ = '\\'+allDir
    filepath_min = os.path.join('%s%s' % (filepath, allDir__))
    filepath_save = os.path.join('%s%s' % ('F:\\natong\\natong_product_skew_crop', allDir__))
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

        image = cv2.imread(filepath_final)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur and threshold the image
        blurred = cv2.blur(gray, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
        # 形态学方面的操作填充内部白色区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        # box = np.int0(cv2.cv.BoxPoints(rect))
        # draw a bounding box arounded the detected barcode and display the image
        # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        cropImg = image[y1:y1 + hight, x1:x1 + width]
        #保存
        filepath_min_save = filepath_save + "\\"+alldir_min
        print(filepath_min_save)
        cv2.imwrite(filepath_min_save, cropImg)