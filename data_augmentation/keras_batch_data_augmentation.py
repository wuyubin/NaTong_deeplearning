# -*- coding: UTF-8 -*-
import os
import math
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
datagen=ImageDataGenerator(
      rotation_range=0,
      width_shift_range=0.1,
      height_shift_range=0.1,
      rescale=1./255,
      shear_range=0,
      zoom_range=0.1,
      horizontal_flip=True,
      fill_mode='nearest',
        cval=0,
        channel_shift_range=0,
        vertical_flip=True)


# 遍历指定目录，显示目录下的所有文件名f
filepath = "F:\\natong\\natong_product_skew_crop"
pathDir = os.listdir(filepath)
for allDir in pathDir:
    allDir__ = '\\'+allDir
    filepath_min = os.path.join('%s%s' % (filepath, allDir__))
    filepath_save = os.path.join('%s%s' % ('F:\\natong\\natong_data_augmentation', allDir__))
    # filepath_save = filepath_save + '_mf'
    print (filepath_save)            # child.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题
    pathdir_min = os.listdir(filepath_min)
    # # 保存路径
    # isExists = os.path.exists(filepath_save)
    # # 判断结果
    # if not isExists:
    #     # 如果不存在则创建目录
    #     os.makedirs(filepath_save)
    count = 0
    for i in pathdir_min:
        if os.path.isfile(os.path.join(filepath_min, i)):
            count += 1
    n = math.floor(190/count)
    for alldir_min in pathdir_min:
        alldir_min__ = '\\' + alldir_min

        filepath_final = os.path.join('%s%s' % (filepath_min, alldir_min__))

        # print(filepath_final)
        read_path = filepath_final
        img=load_img(read_path)
        x=img_to_array(img)
        x=x.reshape((1,)+x.shape)

        i=0
        for batch in datagen.flow(x,batch_size=1,
                         save_to_dir=filepath_save,save_prefix=alldir_min,save_format='bmp'):
            i+=1
            if i>n:
                break




