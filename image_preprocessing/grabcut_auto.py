import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import filters
import time
start = time.clock()

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


name = '00000009_00000000001D10A3'
path = 'F://natong//natong_data_augmentation//13022315000008//auto_grabcut//'+name+'.bmp'
img = cv2.imread(path)


mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

mask_old = cv2.imread(path,0)
img02 = salt(mask_old, 500)
mask_old = cv2.medianBlur(img02, 5)
thresh = filters.threshold_li(mask_old)
newmask =(mask_old <= thresh)*1.0     #阈值化

# cv2.imshow('aaaaaaaaa',newmask)
# cv2.waitKey(0)

# whereever it is marked white (sure foreground), change mask=1
# whereever it is marked black (sure background), change mask=0
# mask[newmask == 0] = 0
# mask[newmask == 255] = 1
mask[newmask == 0] = 1
mask[newmask == 1] = 0
mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
#两种展示方式
# mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask[:,:,np.newaxis]
# plt.imshow(img),plt.colorbar(),plt.show()

mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')

output = cv2.bitwise_and(img,img,mask=mask2)

output = salt(output, 500)
output = cv2.medianBlur(output, 5)
pathth = 'F://natong//natong_data_augmentation//13022315000008//auto_grabcut_result//'+name+'.jpg'
cv2.imwrite(pathth, output)
end = time.clock()
print ("read: %f s" % (end - start))

# plt.imshow(output),plt.colorbar(),plt.show()