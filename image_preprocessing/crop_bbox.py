from PIL import Image
im = Image.open(r"F:\natong\natong_data_augmentation\13022315000008\auto_grabcut_result\1.jpg")
im.show()
tuple = im.getbbox()
print(tuple)
im.crop(tuple)
# im.show()
im.save(r"F:\natong\natong_data_augmentation\13022315000008\auto_grabcut_result\save.JPG")