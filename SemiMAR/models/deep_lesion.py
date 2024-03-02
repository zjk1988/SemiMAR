import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import os
import time
import h5py
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import transforms
from scipy.io import loadmat
import torch.utils.data as Data
from PIL import Image
# import torchvision.transforms as T
import torch.utils.data 
# import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
import torch.distributed as dist
import argparse
# import odl
# from odl.contrib import torch as odl_torch
import random



class DeepLesion(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='data/deep_lesion/train',
        blacklist='data/deep_lesion/blacklist.json', normalize=True, partial_holdout=0,
        hu_offset=32768, random_mask=True, random_flip=False, load_mask=False):
        super(DeepLesion, self).__init__()
        
        # a_dir = "../ma_CT2021"
        c_dir = '/data/tt/data_deep_lesion/gt_CT/'
        d_dir = '/data/tt/data_deep_lesion/gt_CT/'

        # a_dir = '/data/tt/data_deep_lesion/ma_CT/'
        a_dir = '/data/tt/data_deep_lesion/test_ma_CT/'
        # c_dir = '../test_gt_CT'
        # d_dir = '../test_pre_CT2021'
        
        if not os.path.isdir(a_dir):
            raise ValueError("input file_path is not a dir")
        self.a_dir = a_dir
        self.ma_CT = os.listdir(a_dir)
        self.ma_CT.sort()

        # if not os.path.isdir(b_dir):
        #     raise ValueError("input file_path is not a dir")
        # self.b_dir = b_dir
        # self.ma_CTB = os.listdir(b_dir)
        # self.ma_CTB.sort() 

        if not os.path.isdir(c_dir):
            raise ValueError("input file_path is not a dir")
        self.c_dir = c_dir
        self.clean_CT = os.listdir(c_dir)
        self.clean_CT.sort() 

        if not os.path.isdir(d_dir):
            raise ValueError("input file_path is not a dir")
        self.d_dir = d_dir
        self.pre_CT = os.listdir(d_dir)
        self.pre_CT.sort() 

        
        
     
    def __getitem__(self, index):
        # index_B = random.randint(0, 17000)
        index_B = index        
        
        f = os.path.join(self.a_dir, self.ma_CT[index])
        data = loadmat(f)
        data = data['ma_CT']
        # data = h5py.File(f,'r')
        # data = data['ma_CT'][:].T
        data[data<0]=0
        data[data>0.7816]=0.7816
        data = np.expand_dims(data, axis=0)
        A = torch.FloatTensor(data)
       

        data_name = os.path.basename(f)

        f = os.path.join(self.c_dir, self.clean_CT[index_B])
        data = h5py.File(f,'r')
        data = data['gt_CT'][:].T
        data[data<0]=0
        data[data>0.7816]=0.7816
        data = np.expand_dims(data, axis=0)
        C = torch.FloatTensor(data)

        f = os.path.join(self.c_dir, self.pre_CT[index])
        data = h5py.File(f,'r')
        data = data['gt_CT'][:].T
        data[data<0]=0
        data[data>0.7816]=0.7816
        data = np.expand_dims(data, axis=0)
        D = torch.FloatTensor(data)

        # f = os.path.join(self.c_dir, self.clean_CT[index1])
        # data = loadmat(f)
        # data = data['gt_CT']
        # data = np.expand_dims(data, axis=0)
        # C = torch.FloatTensor(data)

        # f = os.path.join(self.c_dir, self.clean_CT[index])
        # data = loadmat(f)
        # data = data['gt_CT']
        # data = np.expand_dims(data, axis=0)
        # D = torch.FloatTensor(data)
        
       
        # return {"data_name": data_name, "a": ma_CT, "b": clean_CT}
        
        return {"data_name": data_name, "A": A, 'C': C, 'D':D}

    def __len__(self):
        # print('self.ma_CT',len(self.ma_CT))
        # return 17000
        return 2000