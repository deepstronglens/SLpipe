from __future__ import division
import os

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


class channel_RGB(object):
    def __init__(self, RED=None, GREEN=None, BLUE=None, backsub=False):
        self.red   = RED
        self.green = GREEN
        self.blue  = BLUE
        self.check_image_shapes()
        if backsub:
            self.subtract_background()

    def check_image_shapes(self):
        if (np.shape(self.red) != np.shape(self.green)) or \
            (np.shape(self.red) != np.shape(self.blue)):
            raise "Image arrays are of different shapes, exiting"
        else:
            self.NX, self.NY = np.shape(self.red)

    def subtract_background(self):
        self.red   -= np.median(self.red)
        self.green -= np.median(self.green)
        self.blue  -= np.median(self.blue)

    def apply_scale(self, scales=(1.0,1.0,1.0)):
        assert len(scales) == 3
        s1,s2,s3 = scales
        mean = (s1 + s2 + s3)/3.0
        self.red   *= (s1/mean)
        self.green *= (s2/mean)
        self.blue  *= (s3/mean)

    def pjm_offset(self, offset=0.0):
        if offset==None:
            pass
        else:
            self.red   += offset
            self.green += offset
            self.blue  += offset

    def pjm_mask(self,masklevel=None):
        if masklevel==None:
            pass
        else:
            tiny = 1e-12
            mask = self.red*0.0 + 1.0
            for image in (self.red, self.green, self.blue):
                image[np.isnan(image)] = 0.0
                image[np.isinf(image)] = 0.0
                mask[image < masklevel] = 0.0
                mask[(image > -tiny) & (image < tiny)] = 0.0
            self.red   *= mask
            self.green *= mask
            self.blue  *= mask

    def lupton_stretch(self, Q=1.0, alpha=1.0, itype='sum'):
        if itype == 'sum':
            I = (self.red+self.green+self.blue) + 1e-10
        elif itype == 'rms':
            I = np.sqrt(self.red**2.0+self.green**2.0+self.blue**2.0) + 1e-10
        stretch = np.arcsinh(alpha*Q*I) / (Q*I)
        self.red   *= stretch
        self.green *= stretch
        self.blue  *= stretch

    def lupton_saturate(self,threshold=1.0, saturation='white', unsat=0.995):
        if saturation=="white":
            pass
        elif saturation=="color":
            x = np.dstack((self.red, self.green,self.blue))
            maxpix = np.max(x, axis=-1)
            maxpix[maxpix<threshold] = 1.0
            self.red   /= maxpix
            self.green /= maxpix
            self.blue  /= maxpix
        else:
            print("Not a recognized type of saturation!!!")

        all_tmp = np.hstack([self.red.ravel(), self.green.ravel(), self.blue.ravel()])
        self.red    /= (all_tmp[all_tmp.argsort()[int(np.round(len(all_tmp)*unsat))]])
        self.green  /= (all_tmp[all_tmp.argsort()[int(np.round(len(all_tmp)*unsat))]])
        self.blue   /= (all_tmp[all_tmp.argsort()[int(np.round(len(all_tmp)*unsat))]])

    def pack_up(self, unsat=0.995):
        x = np.zeros([self.NX,self.NY,3])
        x[:,:,0] = np.flipud(self.red)
        x[:,:,1] = np.flipud(self.green)
        x[:,:,2] = np.flipud(self.blue)
        x = np.clip(x,0.0,1.0)
        x = x*255
        self.imgRGB = Image.fromarray(x.astype(np.uint8))

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)

        def Viz_data(LR_data,HR_data,HR_limg_data,name):

            scales, offset, Q, alpha, masklevel, saturation, itype = (0.9,1.1,1.5), 0.0, 20, 40.8, -1.0, 'color', 'rms'
            
            def Apply_viz_trnsfm(img_i_rscl,img_r_rscl,img_g_rscl):
                object_RGB = channel_RGB(RED=img_i_rscl, GREEN=img_r_rscl, BLUE=img_g_rscl)
                object_RGB.apply_scale(scales=scales)      
                object_RGB.lupton_stretch(Q=Q, alpha=alpha, itype=itype)
                object_RGB.pjm_mask(masklevel=masklevel)     
                object_RGB.pjm_offset(offset=offset)       
                object_RGB.lupton_saturate(saturation=saturation)
                object_RGB.pack_up()    

                return object_RGB.imgRGB        

            fig, ax = plt.subplots(nrows=4, ncols=20, figsize=[50, 10])
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

            for i in range(20):
                imgRGB = Apply_viz_trnsfm(LR_data[i,:,:,2],LR_data[i,:,:,1],LR_data[i,:,:,0])
                img = ax[0,i].imshow(imgRGB)
                ax[0,i].set_title ("LR_i="+str(i))
                ax[0,i].set_xticklabels([])
                ax[0,i].set_yticklabels([])
                ax[0,i].set_aspect('equal')

                imgRGB = Apply_viz_trnsfm(HR_data[i,:,:,2],HR_data[i,:,:,1],HR_data[i,:,:,0])
                img = ax[1,i].imshow(imgRGB)
                ax[1,i].set_title ("HR_i="+str(i))
                ax[1,i].set_xticklabels([])
                ax[1,i].set_yticklabels([])
                ax[1,i].set_aspect('equal')

                imgRGB = Apply_viz_trnsfm(LR_data[i,:,:,2]-HR_data[i,:,:,2],LR_data[i,:,:,1]-HR_data[i,:,:,1],LR_data[i,:,:,0]-HR_data[i,:,:,0])
                img = ax[2,i].imshow(imgRGB)
                ax[2,i].set_title ("LR-HR_i="+str(i))
                ax[2,i].set_xticklabels([])
                ax[2,i].set_yticklabels([])
                ax[2,i].set_aspect('equal')

                imgRGB = Apply_viz_trnsfm(HR_limg_data[i,:,:,2],HR_limg_data[i,:,:,1],HR_limg_data[i,:,:,0])
                img = ax[3,i].imshow(imgRGB)
                ax[3,i].set_title ("HR_limg_i="+str(i))
                ax[3,i].set_xticklabels([])
                ax[3,i].set_yticklabels([])
                ax[3,i].set_aspect('equal')

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(img, cax=cbar_ax)
            plt.savefig('/KDD_Submission/Denoising_Deblending/EDSR_MWCNN/experiment/PlotData/'+name+'.png')

        def Apply_viz_trnsfm(img_i_rscl,img_r_rscl,img_g_rscl):
            scales, offset, Q, alpha, masklevel, saturation, itype = (0.9,1.1,1.5), 0.0, 20, 40.8, -1.0, 'color', 'rms'

            object_RGB = channel_RGB(RED=img_i_rscl, GREEN=img_r_rscl, BLUE=img_g_rscl)
            object_RGB.apply_scale(scales=scales)      
            object_RGB.lupton_stretch(Q=Q, alpha=alpha, itype=itype)
            object_RGB.pjm_mask(masklevel=masklevel)     
            object_RGB.pjm_offset(offset=offset)       
            object_RGB.lupton_saturate(saturation=saturation)
            object_RGB.pack_up()    

            return object_RGB.imgRGB 

        def _load_bin(split):

            if (self.args.denoise):
                print ("Selected the denoising model")
                denoise = True
                deblend = False
                save_name_data = 'denoise'
            else:
                denoise = False
                deblend = True
                save_name_data = 'deblend'

            denoise_and_deblend = False # TODO: automate this

            def Normalization_SL(images_lr,images_hr,images_limg_hr,Calc_max,MaxMinNorm=True):

                print ("orig-images_hr",images_hr.max(),images_hr.min())  
                print ("orig-images_lr[0]",images_lr[0].max(),images_lr[0].min()) 
                print ("orig-images_limg_hr",images_limg_hr.max(),images_limg_hr.min()) 

                if  MaxMinNorm:
                    # Calculate the normalizing factor using {source galaxy+lensed light+noise}
                    lx_min = images_lr[0].min(axis=(1, 2, 3), keepdims=True)
                    lx_max = images_lr[0].max(axis=(1, 2, 3), keepdims=True)

                    # Normalize the {source galaxy+lensed light}
                    images_hr = ((images_hr - lx_min)/(lx_max-lx_min))*255

                    # Normalize the {source galaxy+lensed light+noise}
                    images_lr[0] = ((images_lr[0] - lx_min)*255)/(lx_max-lx_min)

                    # Normalize the {source galaxy}
                    images_limg_hr = ((images_limg_hr - lx_min)/(lx_max-lx_min))*255
                
                else: 
                    # Calculate the normalizing factor using {source galaxy+lensed light+noise}
                    lx_max = Calc_max.max(axis=(1, 2, 3), keepdims=True)

                    print ("lx_max",np.argmax(lx_max),np.argmin(lx_max))

                    # Normalize the {source galaxy+lensed light}
                    images_hr = ((images_hr/lx_max).clip(1e-10,1))#*255

                    # Normalize the {source galaxy+lensed light+noise}
                    images_lr[0] = ((images_lr[0]/lx_max).clip(1e-10,1))#*255

                    # Normalize the {source galaxy}
                    images_limg_hr = ((images_limg_hr/lx_max).clip(1e-10,1))#*255 

                print ("Norm-images_hr",images_hr.max(),images_hr.min())  
                print ("Norm-images_lr[0]",images_lr[0].max(),images_lr[0].min()) 
                print ("Norm-images_limg_hr",images_limg_hr.max(),images_limg_hr.min())             

                return images_lr,images_hr,images_limg_hr


            if (split == 'train'):
                n = 18000
                Calc_max = np.load(self._name_lrbin(1),allow_pickle=True)['X'].transpose(1,2,3,0)[0:n,:,:,:]
                if (denoise):
                    print ("denoise-Training")
                    self.images_hr = np.load(self._name_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[0:n,:,:,:]
                    self.images_limg_hr = np.load(self._name_limg_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[0:n,:,:,:]
                    self.images_lr = [
                        np.load(self._name_lrbin(s),allow_pickle=True)['X'].transpose(1,2,3,0)[0:n,:,:,:] for s in self.scale
                    ]
                    self.images_hr = self.images_lr[0] - self.images_hr

                elif (deblend):
                    self.images_hr = np.load(self._name_limg_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[0:n,:,:,:]
                    self.images_lr = [
                        (np.load(self._name_lrbin(s),allow_pickle=True)['X'].transpose(1,2,3,0)[0:n,:,:,:]-
                        np.load(self._name_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[0:n,:,:,:]) for s in self.scale
                    ]                    
                    self.images_limg_hr = np.load(self._name_limg_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[0:n,:,:,:]

                elif (denoise_and_deblend):
                    self.images_lr = [
                        np.load(self._name_lrbin(s),allow_pickle=True)['X'].transpose(1,2,3,0)[0:n,:,:,:] for s in self.scale
                    ]  
                    self.images_hr = np.load(self._name_limg_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[0:n,:,:,:] 
                    self.images_limg_hr = self.images_hr      
                             
                ################ Moving the Normalization and transformation code to dataloader function to apply this transformation batchwise
                self.images_lr,self.images_hr,self.images_limg_hr = Normalization_SL(self.images_lr,self.images_hr,self.images_limg_hr,Calc_max,MaxMinNorm=False)
                
                for i in range (len(self.images_hr)):
                    self.images_lr[0][i,:,:,:] = Apply_viz_trnsfm(self.images_lr[0][i,:,:,2],self.images_lr[0][i,:,:,1],self.images_lr[0][i,:,:,0])
                    self.images_hr[i,:,:,:] = Apply_viz_trnsfm(self.images_hr[i,:,:,2],self.images_hr[i,:,:,1],self.images_hr[i,:,:,0])
                    self.images_limg_hr[i,:,:,:] = Apply_viz_trnsfm(self.images_limg_hr[i,:,:,2],self.images_limg_hr[i,:,:,1],self.images_limg_hr[i,:,:,0])
                
                print ("Viz-Norm-images_hr",self.images_hr.max(),self.images_hr.min(),'shape',self.images_hr.shape)  
                print ("Viz-Norm-images_lr[0]",self.images_lr[0].max(),self.images_lr[0].min()) 
                print ("Viz-Norm-images_limg_hr",self.images_limg_hr.max(),self.images_limg_hr.min())
                ###################################################               

                
            else:
                " Data preparation for testing in between the training epochs or for inference with the pretrained network"
                st = self.args.test_st  
                ed = self.args.test_end 

                st_dl = self.args.test_st 
                ed_dl = self.args.test_end
                inference_pipeline = self.args.inference_pipeline  
                
                if (denoise):
                    print ("Test - Denoising", "total data=",ed-st, "inference_pipeline=",inference_pipeline )
                    Calc_max = np.load(self._name_lrbin(1),allow_pickle=True)['X'].transpose(1,2,3,0)[st:ed,:,:,:]
                    "With the denoising dataset, the images_hr is only the noise and image_lr is Clean image+noise" 
                    "finally image_hr is prepared by subtracting both"
                    self.images_hr = np.load(self._name_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[st:ed,:,:,:]
                    self.images_limg_hr = np.load(self._name_limg_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[st:ed,:,:,:]
                    self.images_lr = [
                        np.load(self._name_lrbin(s),allow_pickle=True)['X'].transpose(1,2,3,0)[st:ed,:,:,:] for s in self.scale
                    ]
                    self.images_hr = self.images_lr[0] - self.images_hr
                elif (deblend):
                    "With the deblending dataset, the images_hr is limg_hrbin and image_lr is _name_lrbin - _name_hrbin" 
                    "image_hr is limg which is the clean image without lensed light, image_lr is the clean image" 
                        
                    if  (inference_pipeline): 
                        "read the .npz file that was the SR output from denoising applied on 120kdata for inference"
                        " This is read as LR into inference with the deblending model"
                        print ("LR data input to the Deblending model inference - obtained from SR of the denoising model")
                        self.images_lr = [np.load(self._name_lrbin(s).rsplit('.',1)[0]+'_DL.npz',allow_pickle=True)['X'].transpose(0,2,3,1)[st_dl:ed_dl,:,:,:] for s in self.scale]
                        self.images_hr = np.load(self._name_hrbin().rsplit('.',1)[0]+'_DL.npz',allow_pickle=True)['X'].transpose(0,2,3,1)[st_dl:ed_dl,:,:,:]
                        self.images_limg_hr = self.images_hr

                        print ("shape",self.images_hr.shape, self.images_lr[0].shape)
                        print ("Max_HR",self.images_hr.max())
                        print ("Min_HR",self.images_hr.min())
                        print ("Max_LR",self.images_lr[0].max())
                        print ("Min_LR",self.images_lr[0].min())

                    else:
                        Calc_max = np.load(self._name_lrbin(1),allow_pickle=True)['X'].transpose(1,2,3,0)[st:ed,:,:,:]
                        " This is the case where the LR is taken from the simulation data (fimg-noise)"
                        print ("LR data input to the Deblending model inference - taken from the simulation data (fimg-noise)")
                        self.images_lr = [
                            (np.load(self._name_lrbin(s),allow_pickle=True)['X'].transpose(1,2,3,0)[st:ed,:,:,:]-
                            np.load(self._name_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[st:ed,:,:,:]) for s in self.scale
                        ] 
                        self.images_hr = np.load(self._name_limg_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[st:ed,:,:,:]
                   
                        self.images_limg_hr = np.load(self._name_limg_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[st:ed,:,:,:]   

                elif (denoise_and_deblend):
                    self.images_lr = [
                        np.load(self._name_lrbin(s),allow_pickle=True)['X'].transpose(1,2,3,0)[st:ed,:,:,:] for s in self.scale
                    ]  
                    self.images_hr = np.load(self._name_limg_hrbin(),allow_pickle=True)['X'].transpose(1,2,3,0)[st:ed,:,:,:] 
                    self.images_limg_hr = self.images_hr 
                
                #########################################################
                if not inference_pipeline:

                    self.images_lr,self.images_hr,self.images_limg_hr = Normalization_SL(self.images_lr,self.images_hr,self.images_limg_hr,Calc_max,MaxMinNorm=False)

                    for i in range (len(self.images_hr)):
                        self.images_lr[0][i,:,:,:] = Apply_viz_trnsfm(self.images_lr[0][i,:,:,2],self.images_lr[0][i,:,:,1],self.images_lr[0][i,:,:,0])
                        self.images_hr[i,:,:,:] = Apply_viz_trnsfm(self.images_hr[i,:,:,2],self.images_hr[i,:,:,1],self.images_hr[i,:,:,0])
                        self.images_limg_hr[i,:,:,:] = Apply_viz_trnsfm(self.images_limg_hr[i,:,:,2],self.images_limg_hr[i,:,:,1],self.images_limg_hr[i,:,:,0])
                #########################################################

        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr = self._scan()
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    hr = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for si, s in enumerate(self.scale):
                    for v in self.images_lr[si]:
                        lr = misc.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]

        elif args.ext.find('bin') >= 0:
            print('Loading a binary file')
            _load_bin(self.split)
        else:
            print('Please define data type')




    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):
        lr, hr, lr2, hr2, filename = self._load_file(idx)

        lr, hr, lr2, hr2 = self._get_patch(lr, hr, lr2, hr2)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr2, hr2 = common.set_channel([lr2, hr2], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        lr_tensor2, hr_tensor2 = common.np2Tensor([lr2, hr2], self.args.rgb_range)
        return lr_tensor, hr_tensor, lr_tensor2, hr_tensor2, filename

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        lr2 = hr
        hr2 = self.images_limg_hr[idx]

        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = misc.imread(lr)
            hr = misc.imread(hr)
            print ("np.shape(lr)",np.shape(lr),lr.dtype)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, lr2, hr2, filename

    def _get_patch(self, lr, hr,lr2=None,hr2=None):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1

        if self.train:
            if (self.args.nmodels == 1):
                lr, hr = common.get_patch(
                    lr, hr, patch_size, scale, multi_scale=multi_scale
                )
                lr, hr = common.augment([lr, hr])
                lr = common.add_noise(lr, self.args.noise)
            else:
                lr, hr, lr2, hr2 = common.get_patch_2model(
                    lr, hr, lr2, hr2, patch_size, scale, multi_scale=multi_scale
                )               
        else:
            if (self.args.nmodels == 1):
                ih, iw = lr.shape[0:2]
                hr = hr[0:ih * scale, 0:iw * scale]
            else:
                ih, iw = lr.shape[0:2]
                hr = hr[0:ih * scale, 0:iw * scale]            
                hr2 = hr2[0:ih * scale, 0:iw * scale]   

        return lr, hr, lr2, hr2

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
