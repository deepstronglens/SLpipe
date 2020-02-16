import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
import glob

class Sim_data(srdata.SRData):
    def __init__(self, args, train=True):
        super(Sim_data, self).__init__(args, train)
        self.repeat = 1#args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        print ("Scan module")
        if self.train:
	    # extract the indices of the training data	
            list_hr = glob.glob(self.dir_hr_train+'/*'+self.ext)
            list_lr0 = glob.glob(self.dir_lr_train+'/*'+self.ext)
            list_hr_aug = glob.glob(self.dir_hr_train_aug+'/*'+self.ext)
            list_lr0_aug = glob.glob(self.dir_lr_train_aug+'/*'+self.ext)
            list_hr = list_hr + list_hr_aug
            list_lr0 = list_lr0 + list_lr0_aug
            for si, s in enumerate(self.scale):
                   list_lr[si].extend(list_lr0)
            print ("train_length",len(list_hr))
        else:
	    # Extract the indices of the test and val data
            print ("HR path",self.dir_hr_val)
            print ("LR path",self.dir_lr_val)
            list_hr_v = glob.glob(self.dir_hr_val+'/*'+self.ext)
            list_lr_v = glob.glob(self.dir_lr_val+'/*'+self.ext)
            list_hr = list_hr_v 
            list_lr0 = list_lr_v

            for si, s in enumerate(self.scale):
                   list_lr[si].extend(list_lr0)
            print ("len(list_hr),len(list_lr)",len(list_hr),len(list_lr))
        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        self.dir_hr_val = os.path.join(self.apath, 'Source_galaxy')
        self.dir_lr_val = os.path.join(self.apath, 'Combined_LR')
        self.ext = '.png'

    def _name_hrbin(self,saved=None):
        if saved is None:
            return os.path.join(
                self.apath,
                'bin',
                '{}_bin_HR.npz'.format(self.split)
            )
        else:
            return os.path.join(
                self.apath,
                'bin',
                '{}_{}_bin_HR.npz'.format(self.split,saved)
            )        

    def _name_lrbin(self, scale, saved=None):
        if saved is None:
            return os.path.join(
                self.apath,
                'bin',
                #'{}_bin_LR.npy'.format(self.split)
                '{}_bin_LR_X{}.npz'.format(self.split, scale)
            )
        else:
            return os.path.join(
                self.apath,
                'bin',
                #'{}_bin_LR.npy'.format(self.split)
                '{}_{}_bin_LR_X{}.npz'.format(self.split,saved,scale)
            )
    def _name_limg_hrbin(self,saved=None):
        if saved is None:
            return os.path.join(
                self.apath,
                'bin',
                '{}_limg_bin_HR.npz'.format(self.split)
            )
        else:
            return os.path.join(
                self.apath,
                'bin',
                '{}_{}_limg_bin_HR.npz'.format(self.split, saved)
            )            

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

