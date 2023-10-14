import os
import numpy as np
import pandas as pd
import multiprocessing

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.utils import shuffle

import cv2
import glob


path = 'C:/Users/Siryu_sci/2021-MLVU/SLT_project/features/fullFrame-210x260px/'

def save_video(folder):
    storage = 'C:/Users/Siryu_sci/2021-MLVU/SLT_project/frames/'

    name = folder
    imgpath = path+folder

    if not os.path.exists(storage + name + '.pt'):

        images = torch.tensor(np.array([cv2.imread(file) for file in glob.glob(imgpath + '/*.png')]))

        T, H, W, C = images.shape
        if images.shape[0] is not None:
            images = images.permute(0, 3, 1, 2)
            images = images.type(torch.uint8)
            print(name, '\t', images.shape)
            torch.save(images, storage + name + '.pt')


folder_list = os.listdir(path)
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    pool.map(save_video, folder_list)
    pool.close()
    pool.join()