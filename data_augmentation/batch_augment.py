# -*- coding:utf-8 -*-
"""数据增强
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection

"""

from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging
import math

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# def rotate(image, angle, center=None, scale=1.0):
#     # 获取图像尺寸
#     (h, w) = image.shape[:2]
#     # 若未指定旋转中心，则将图像中心设为旋转中心
#     if center is None:
#         center = (w / 2, h / 2)
#     # 执行旋转
#     M = cv2.getRotationMatrix2D(center, angle, scale)
#     rotated = cv2.warpAffine(image, M, (w, h))
#     # 返回旋转后的图像
#     return rotated

class DataAugmentation:
    """
    包含数据增强的八种方式
    """


    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        # return image.rotate(random_angle, mode)
        return image.rotate(random_angle)
        # return rotate(image, random_angle)

    @staticmethod
    def transpose_left_right(image):
        """
         对图像进行反转
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    @staticmethod
    def transpose_top_bottom(image):
        """
         对图像进行反转
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    @staticmethod
    def randomCrop(image):
        """
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像

        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(478, 479)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        # return image.crop(random_region)
        # 加载底图
        base_img = Image.open('background.bmp')
        (X1, Y1) = base_img.size
        # 加载需要P上去的图片
        tmp_img = image.crop(random_region)
        # 这里可以选择一块区域或者整张图片
        region = tmp_img
        (x, y) = tmp_img.size
        xx = int((X1 - x))
        xx = random.randint(1, xx)
        yy = int((Y1 - y))
        yy = random.randint(1, yy)
        box = (xx, yy, xx + x, yy + y)  # 底图上需要P掉的区域
        # 使用 paste(region, box) 方法将图片粘贴到另一种图片上去.
        # 注意，region的大小必须和box的大小完全匹配。但是两张图片的mode可以不同，合并的时候回自动转化。如果需要保留透明度，则使用RGMA mode
        # 提前将图片进行缩放，以适应box区域大小
        # region = region.rotate(180) #对图片进行旋转
        region = region.resize((box[2] - box[0], box[3] - box[1]))
        base_img.paste(region, box)
        random_angle = np.random.randint(1, 360)
        return base_img.rotate(random_angle)


    @staticmethod
    def resize_by_scale(image):
        """按照所需比例缩放"""
        im = image
        (x, y) = im.size
        scale = random.randint(430, 510)
        if x > y:
            x_s = scale
            a = math.ceil(x / scale)
            y_s = math.floor(y / a)
        else:
            y_s = scale
            b = math.ceil(y / scale)
            x_s = math.floor(x / b)
        out = im.resize((x_s, y_s), Image.ANTIALIAS)
        # 加载底图
        base_img = Image.open('background.bmp')
        (X1, Y1) = base_img.size
        # 加载需要P上去的图片
        tmp_img = out
        # 这里可以选择一块区域或者整张图片
        region = tmp_img
        (x, y) = tmp_img.size
        xx = int((X1 - x))
        xx = random.randint(1, xx)
        yy = int((Y1 - y))
        yy = random.randint(1, yy)
        box = (xx, yy, xx + x, yy + y)  # 底图上需要P掉的区域
        # 使用 paste(region, box) 方法将图片粘贴到另一种图片上去.
        # 注意，region的大小必须和box的大小完全匹配。但是两张图片的mode可以不同，合并的时候回自动转化。如果需要保留透明度，则使用RGMA mode
        # 提前将图片进行缩放，以适应box区域大小
        # region = region.rotate(180) #对图片进行旋转
        region = region.resize((box[2] - box[0], box[3] - box[1]))
        base_img.paste(region, box)
        random_angle = np.random.randint(1, 360)
        return base_img.rotate(random_angle)

    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        base_img = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
        random_angle = np.random.randint(1, 360)
        return base_img.rotate(random_angle)

    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def saveImage(image, path):
        image.save(path)


def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception as e:
        print (str(e))
        return -2


def imageOps(image, des_path, file_name):
    """
    :param func_name:方法名
    :param image: 输入图像
    :param des_path: 存储路径
    :param file_name: 文件名
    :param times: 调用次数
    :return:
    """
    funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomCrop": DataAugmentation.randomCrop,
               "randomColor": DataAugmentation.randomColor,
               "randomGaussian": DataAugmentation.randomGaussian,
               "resize_by_scale":DataAugmentation.resize_by_scale,
               "transpose_left_right":DataAugmentation.transpose_left_right,
               "transpose_top_bottom":DataAugmentation.transpose_top_bottom
               }
    # if funcMap.get(func_name) is None:
    #     logger.error("%s is not exist", func_name)
    #     return -1
    if "transpose_top_bottom" in funcMap.keys():
        for _i in range(0, 1, 1):
            new_image = DataAugmentation.transpose_top_bottom(image)
            DataAugmentation.saveImage(new_image, os.path.join(des_path, "transpose_top_bottom" + str(_i) + file_name))
    if "transpose_left_right" in funcMap.keys():
        for _i in range(0, 1, 1):
            new_image = DataAugmentation.transpose_left_right(image)
            DataAugmentation.saveImage(new_image, os.path.join(des_path, "transpose_left_right" + str(_i) + file_name))
    if "randomRotation" in funcMap.keys():
        for _i in range(0, 17, 1):
            new_image = DataAugmentation.randomRotation(image)
            DataAugmentation.saveImage(new_image, os.path.join(des_path, "randomRotation" + str(_i) + file_name))
    if "randomCrop" in funcMap.keys():
        for _i in range(0, 5, 1):
            new_image = DataAugmentation.randomCrop(image)
            DataAugmentation.saveImage(new_image, os.path.join(des_path, "randomCrop" + str(_i) + file_name))
    if "randomColor" in funcMap.keys():
        for _i in range(0, 6, 1):
            new_image = DataAugmentation.randomColor(image)
            DataAugmentation.saveImage(new_image, os.path.join(des_path, "randomColor" + str(_i) + file_name))
    if "resize_by_scale" in funcMap.keys():
        for _i in range(0, 13, 1):
            new_image = DataAugmentation.resize_by_scale(image)
            DataAugmentation.saveImage(new_image, os.path.join(des_path, "resize_by_scale"  + str(_i) + file_name))

    # for _i in range(0, times, 1):
    #     new_image = funcMap[func_name](image)
    #     DataAugmentation.saveImage(new_image, os.path.join(des_path, func_name + str(_i) + file_name))


opsList = {"randomRotation", "randomCrop", "randomColor","resize_by_scale","transpose_left_right","transpose_top_bottom"}

if __name__ == '__main__':
    # 遍历指定目录，显示目录下的所有文件名
    filepath = "F:\\natong\\add"
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        allDir__ = '\\' + allDir
        filepath_min = os.path.join('%s%s' % (filepath, allDir__))
        filepath_save = os.path.join('%s%s' % ('F:\\natong\\data_augmentation',allDir__))

        print(filepath_min+'====='+filepath_save)

        # # 保存路径
        # isExists = os.path.exists(filepath_save)
        # # 判断结果
        # if not isExists:
        #     # 如果不存在则创建目录
        #     os.makedirs(filepath_save)

        pathdir_min = os.listdir(filepath_min)
        for alldir_min in pathdir_min:
            alldir_min__ = '\\' + alldir_min
            filepath_final = os.path.join('%s%s' % (filepath_min, alldir_min__))
            image = DataAugmentation.openImage(filepath_final)
            # imageOps(image, filepath_save, alldir_min)