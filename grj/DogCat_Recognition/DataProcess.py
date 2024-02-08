import os
import random
import torchvision.transforms as transforms
from PIL import Image
import multiprocessing as mp
import re

training_size = 11800
# 目标分辨率
target_width, target_height = 128, 128

Dog_Root = '.\\PetImages\\Dog\\'
Cat_Root = '.\\PetImages\\Cat\\'

def Image_Transform(path):
    img = Image.open(path).convert('L')

    # 计算裁剪大小
    aspect_ratio = target_width / target_height
    width, height = img.size
    if width / height < aspect_ratio:
        # 图像过高，按宽度裁剪
        crop_height = width / aspect_ratio
        crop_size = (width, int(crop_height))
    else:
        # 图像过宽，按高度裁剪
        crop_width = height * aspect_ratio
        crop_size = (int(crop_width), height)

    # 创建transform操作
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),  # 居中裁剪
        transforms.Resize((target_width, target_height)),  # 调整分辨率
    ])

    # 应用transform
    img_transformed = transform(img)

    return path, img_transformed

if __name__ == '__main__':
    Dog_list = [Dog_Root + i for i in os.listdir(Dog_Root)[0:-1]]
    Cat_list = [Cat_Root + i for i in os.listdir(Cat_Root)[0:-1]]
    random.shuffle(Dog_list)
    random.shuffle(Cat_list)
    Training = Dog_list[0:training_size]
    Training.extend(Cat_list[0:training_size])
    Test = Dog_list[training_size:]
    Test.extend(Cat_list[training_size:])
    random.shuffle(Training)
    random.shuffle(Test)

    pool = mp.Pool(28)
    Training_img = pool.map(Image_Transform, Training)
    Test_img = pool.map(Image_Transform, Test)
    pool.close()

    for i in Training_img:
        label = re.findall(r'.\\PetImages\\(\w+)',i[0])[0]
        index = re.findall(r'\\(\d+\.jpg)',i[0])[0]
        i[1].save('.\\Training_Data\\'+label+'_'+index)

    for j in Test_img:
        label = re.findall(r'.\\PetImages\\(\w+)',j[0])[0]
        index = re.findall(r'\\(\d+\.jpg)',j[0])[0]
        j[1].save('.\\Test_Data\\'+label+'_'+index)