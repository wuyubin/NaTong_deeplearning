from skimage import data,filters
import matplotlib.pyplot as plt
import cv2
import numpy as np

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

image = cv2.imread(r"F:\natong\natong_data_augmentation\natong_data_augment\12110126300023\00000074_00000000008C3860.bmp_0_6275.jpg",1)

image = salt(image, 500)
image = cv2.medianBlur(image, 5)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = filters.threshold_otsu(image)   #返回一个阈值
# thresh = filters.threshold_yen(image)
# thresh = filters.threshold_li(image)
print(thresh)
dst =(image <= thresh)*1.0   #根据阈值进行分割
img = cv2.resize(dst,(678,444))
cv2.imshow('aaaaaa',img)
cv2.imwrite('1.png',img)
cv2.waitKey(0)
plt.figure('thresh',figsize=(8,8))

plt.subplot(121)
plt.title('original image')
plt.imshow(image,plt.cm.gray)

plt.subplot(122)
plt.title('binary image')
plt.imshow(dst,plt.cm.gray)

plt.show()