# coding=utf-8
from PIL import Image
import math
infile = 'D:\\Python_all\\pythonholder\\practice_project\\NaTong\\NaTong_deeplearning\\image_preprocessing\\contoursImage2.jpg'
outfile = 'D:\\Python_all\\pythonholder\\practice_project\\NaTong\\NaTong_deeplearning\\image_preprocessing\\cont1111oursImage2.jpg'

def resize_by_width(infile, outfile):
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
    out.save(outfile)
resize_by_width(infile,outfile)