#coding=utf-8
import cv2
import numpy as np
img = cv2.imread(r'F:\natong\natong_data_augmentation\13022315000008\auto_grabcut_result\00000009_00000000001D10A3.bmp',1)
#OpenCV定义的结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
#腐蚀图像
eroded = cv2.erode(img,kernel)
#显示腐蚀后的图像
eroded = cv2.resize(eroded,(860,540))
cv2.imshow("Eroded Image",eroded);
#膨胀图像
dilated = cv2.dilate(img,kernel)
#显示膨胀后的图像
dilated = cv2.resize(dilated,(860,540))
cv2.imshow("Dilated Image",dilated);

#NumPy定义的结构元素
NpKernel = np.uint8(np.ones((3,3)))
Nperoded = cv2.erode(img,NpKernel)
#显示腐蚀后的图像
Nperoded = cv2.resize(Nperoded,(860,540))
cv2.imshow("Eroded by NumPy kernel",Nperoded);

#原图像
img = cv2.resize(img,(860,540))
cv2.imshow("Origin", img)
cv2.waitKey(0)
cv2.destroyAllWindows()