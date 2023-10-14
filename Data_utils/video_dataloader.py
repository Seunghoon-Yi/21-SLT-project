import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import os


def get_category(datapath):
    Category = os.listdir(datapath)
    #print(Category, len(Category))
    total_path  = []
    total_label = []

    for lab in Category:
        for sub_lab in os.listdir(datapath + lab + '/'):
            total_label.append(lab)
            total_path.append(datapath + lab + '/' + sub_lab)

    return total_path, total_label, Category



class UCF_dataset(Dataset):
    def __init__(self, total_path, total_label, Category, n_sample = 16, transform = None):
        self.total_path  = total_path
        self.total_label = total_label
        self.category    = Category
        self.n_sample    = n_sample
        self.transform   = transform

    def __len__(self):
        return len(self.total_label)

    def __getitem__(self, index):
        label     = self.total_label[index]
        label     = self.category.index(label)
        VideoPath = self.total_path[index]

        #print(index)

        Video = torchvision.io.read_video(VideoPath)

        SampledVideo = []
        len = Video[0].shape[0]
        for sampleIdx in range(self.n_sample):
            if self.transform is not None:
                frame = torch.tensor(Video[0][len//self.n_sample * sampleIdx]).type(torch.float32)
                frame = self.transform(frame.permute(2,0,1)).permute(1,2,0)
            else:
                frame = torch.tensor(Video[0][len//self.n_sample * sampleIdx]).type(torch.float32)
            SampledVideo.append(frame.unsqueeze(0))

        SampledVideo = torch.cat(SampledVideo, dim = 0)
        #print(len, SampledVideo.shape, SampledVideo.type)


        return SampledVideo, torch.tensor(label)




def main():
    datapath = './UCF_Data/UCF-101/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_path, total_label, Category = get_category(datapath)
    print(len(total_path))

    transforms_ = torch.nn.Sequential(
        transforms.RandomCrop((240, 288)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3))
    )

    UCFDataset = UCF_dataset(total_path, total_label, Category, transform=transforms_)

    UCFDataloader = DataLoader(UCFDataset, batch_size=32, num_workers=0, shuffle=True, pin_memory=True)

    for idx, (VideoBatch, Labels) in enumerate(UCFDataloader):
        VideoBatch = VideoBatch.to(device)
        Labels     = Labels.to(device)
        print(VideoBatch.shape, Labels)

if __name__  == "__main__":
    main()







#sample = torchvision.io.read_video(total_path[123])
#print(sample[0].shape, sample[1].shape)