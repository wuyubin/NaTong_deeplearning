# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
from matplotlib import pyplot
import math

# 定义旋转rotate函数
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    # 返回旋转后的图像
    return rotated

# 遍历指定目录，显示目录下的所有文件名
filepath = "E:\\lenovo_exercitation\\natong_work\\natong_product\\natong_product_grabcut"
pathDir = os.listdir(filepath)
for allDir in pathDir:
    allDir__ = '\\'+allDir
    filepath_min = os.path.join('%s%s' % (filepath, allDir__))
    # print (filepath_save)            # child.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题
    pathdir_min = os.listdir(filepath_min)
    # # 保存路径
    filepath_save = os.path.join('%s%s' % ("F:\\natong\\grabcut_grabcut_skew", allDir__))
    # print(filepath_save)
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply edge detection method on the image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # This returns an array of r and theta values
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)
        # print(lines)
        # The below for loop runs till r and theta values

        # are in the range of the 2d array
        if lines is None:
            print(filepath_final)
            continue
        for r, theta in lines[0]:
            # Stores the value of cos(theta) in a
            a = np.cos(theta)
            # Stores the value of sin(theta) in b
            b = np.sin(theta)
            # x0 stores the value rcos(theta)
            x0 = a * r
            # y0 stores the value rsin(theta)
            y0 = b * r
            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))
            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000 * (a))
            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000 * (-b))
            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000 * (a))
            if x2!=x1:
                jiaodu = math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi
            else :
                jiaodu=90
            # print(jiaodu)
            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be
            # drawn. In this case, it is red.
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        rotated = rotate(img, jiaodu)

        #保存
        filepath_min_save = filepath_save + alldir_min__
        print(filepath_min_save)
        isExists = os.path.exists(filepath_min_save)
        # 判断结果
        if not isExists:
            # 如果保存
            cv2.imwrite(filepath_min_save,rotated, params=None)
