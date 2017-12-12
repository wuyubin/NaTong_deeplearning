# -*- coding: UTF-8 -*-
import os
import math
from PIL import Image

def resize(infile):
    """按照宽度进行所需比例缩放"""
    im = Image.open(infile)
    (x, y) = im.size
    if x>y:
        x_s = 460
        a = math.ceil(x / 460)
        y_s = math.floor(y / a)
    else:
        y_s = 460
        b = math.ceil(y / 460)
        x_s = math.floor(x / b)
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    # out.save(outfile)
    return out
# 遍历指定目录，显示目录下的所有文件名
filepath = "F:\\natong\\natong_product_skew_crop"
pathDir = os.listdir(filepath)
for allDir in pathDir:
    allDir__ = '\\'+allDir
    filepath_min = os.path.join('%s%s' % (filepath, allDir__))
    filepath_save = os.path.join('%s%s' % ('F:\\natong\\natong_product_skew_crop_resize_imagecopy', allDir__))
    # filepath_save = filepath_save + '_mf'
    print (filepath_save)            # child.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题
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

        resized_image = resize(filepath_final)
        # 加载底图
        base_img = Image.open('background.bmp')
        (X1, Y1) = base_img.size
        # 加载需要P上去的图片
        tmp_img = resized_image
        # 或者使用整张图片
        region = tmp_img
        (x, y) = tmp_img.size
        xx = int((X1 - x) / 2)
        yy = int((Y1 - y) / 2)
        box = (xx, yy, xx + x, yy + y)  # 底图上需要P掉的区域
        # 使用 paste(region, box) 方法将图片粘贴到另一种图片上去.
        # 注意，region的大小必须和box的大小完全匹配。但是两张图片的mode可以不同，合并的时候回自动转化。如果需要保留透明度，则使用RGMA mode
        # 提前将图片进行缩放，以适应box区域大小
        # region = region.rotate(180) #对图片进行旋转
        region = region.resize((box[2] - box[0], box[3] - box[1]))
        base_img.paste(region, box)
        # base_img.show()  # 查看合成的图片

        #保存
        filepath_min_save = filepath_save + "\\"+alldir_min
        # print(filepath_min_save)
        isExists = os.path.exists(filepath_min_save)
        # 判断结果
        if not isExists:
            # 如果不存在则创建
            base_img.save(filepath_min_save)  # 保存图片
