import numpy as np
import torch
import os
import math
import random
import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
def splitimage(img_name,new_img_path,patch_size):
    img=Image.open(img_name)
    w,h=img.size
    w_new=(w//patch_size)*patch_size
    h_new=(h//patch_size)*patch_size   #得到整数个新的图块大小
    s_img=os.path.split(img_name)   #分开路径和文件名
    num=30
    rowheight=patch_size  #path_size 代表新的图像的长宽高
    colwidth=patch_size
    for i in range(h_new//patch_size):    #h代表高度
        for j in range(w_new//patch_size):   #w代表宽度
            box = (j * colwidth, i * rowheight, (j + 1) * colwidth, (i + 1) * rowheight)
            img_new_path=os.path.join(new_img_path,'_'+str(num))
            img.crop(box).save(img_new_path+'.png')            #储存分块后的头像
            num=num+1
if __name__ == '__main__':
    path = r'F:\1'
    new_img_path = r'F:\1\image1'
    a =r'F:\1\image1\_22.png'
    splitimage(a, new_img_path, 224)
