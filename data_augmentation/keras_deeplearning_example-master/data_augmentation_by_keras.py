from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
datagen=ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.1,
      height_shift_range=0.1,
      rescale=1./255,
      shear_range=0.1,
      zoom_range=0.1,
      horizontal_flip=True,
      fill_mode='constant',
        cval=0,
        channel_shift_range=0,
        vertical_flip=True)
img=load_img(r'F:\natong\augmenttation--test\test\out02.png')
x=img_to_array(img)
x=x.reshape((1,)+x.shape)

i=0
for batch in datagen.flow(x,batch_size=1,
                         save_to_dir=r'F:\natong\augmenttation--test\train',save_prefix='cat',save_format='jpg'):
    i+=1
    if i>7:
       break
