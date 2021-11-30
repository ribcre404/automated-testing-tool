from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    概率：操作将被执行的概率。
     sl：最小擦除区域
     sh：最大擦除区域
     r1：最小纵横比
     mean：擦除值
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl  # 初始化可选择的擦除区域面积设置的最小比例
        self.sh = sh  # 初始化可选择的擦除区域面积设置的最大比例
        self.r1 = r1  # 初始化最小长宽比
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        '''
        as the selected rectangle region.
        Otherwise repeat the above process until an appropriate Ie
        is selected. With the selected erasing region Ie, each pixel
        in Ie is assigned to a random value in [0, 255], respectively.
        The procedure of selecting the rectangle area and erasing
        this area is shown in Alg. 1.
        '''

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area # 面积范围[sl, sh]*area=[0.02*area, 0.4*area]
            aspect_ratio = random.uniform(self.r1, 1/self.r1)# 长宽比范围[r1, 1/r1]=[0.3. 3.333...]

            h = int(round(math.sqrt(target_area * aspect_ratio))) # 擦除区域高度(height)
            w = int(round(math.sqrt(target_area / aspect_ratio))) # 擦除区域宽度(width)

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h) # 随机起始点纵坐标（height方向）
                y1 = random.randint(0, img.size()[2] - w) # 随机起始点横坐标（width方向）
                if img.size()[0] == 3: # RGB图
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0] # 将擦除区域全部赋予预设小数
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else: # 灰度图
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

