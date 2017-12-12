#coding = 'utf-8'

import cv2
vc = cv2.VideoCapture(r'F:\baidudownload\jingdong_bigdata\Pig_Identification_Qualification_Train\train\1.mp4') #读入视频文件
if vc.isOpened(): #判断是否正常打开
    rval , frame = vc.read()
else:
    rval = False
c=0
timeF = 5  #视频帧计数间隔频率
while rval:   #循环读取视频帧
    rval, frame = vc.read()
    if(c%timeF == 0): #每隔timeF帧进行存储操作
        cv2.imwrite('F:/baidudownload/jingdong_bigdata/Pig_Identification_Qualification_Train/train/31/'+str(c) + '.jpg',frame) #存储为图像
    c = c + 1
    cv2.waitKey(1)
print(c)
vc.release()