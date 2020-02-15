import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import glob
#from netCDF4 import Dataset
import h5py
import pickle


class StrongLensing_new(object):

    def __init__(self, dataset_zip=None):

        # Simulation data
        # loc = '/KDD_Submission/Data/Array_HR.npz'

        # Inference pipeline
        loc = "/KDD_Submission/Denoising_Deblending/EDSR_MWCNN/experiment/Joint_model_inference/Array_SR_2.npz"

        #loc = "/gpfs/jlse-fs0/users/sand33p/Strong_Lensing_Nan/New_Data/Data_XYlabel_fimg_120_new.npz" # baseline
        lab_loc = '/KDD_Submission/Data/Data_Ylabel_fimg_120_new.npz'

            
        def load_data(path_X,path_Y,n_data):
            """Loads a dataset.
            # Arguments
                path: 
            # Returns
                Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
            """

            with np.load(path_X, allow_pickle=True) as f:
                X = f['X']
                print ("input X shape",X.shape)
                X = X#.transpose(1,0,2,3) # use the transpose when loading from Data_XYlabel_fimg_120_new.npz
                st1_x = 0 
                ed1_x = 54000 
                st2_x = 54000 
                ed2_x = 60000 
                x_train = np.concatenate([X[st1_x:ed1_x],X[60000:114000]],axis=0)
                x_test = np.concatenate([X[st2_x:ed2_x],X[114000:120000]],axis=0)      

            with np.load(path_Y, allow_pickle=True) as f:
                Y,l_p = f['label'],f['Y']
                Y = Y#.transpose(1,0)
                l_p = l_p.transpose(1,0)
                st1_x = 0 
                ed1_x = 54000
                st2_x = 54000 
                ed2_x = 60000 
                y_train = np.expand_dims(np.concatenate([Y[st1_x:ed1_x],Y[60000:114000]],axis=0),axis=1)
                y_test = np.expand_dims(np.concatenate([Y[st2_x:ed2_x],Y[114000:120000]],axis=0),axis=1)

                l_p_train = np.concatenate([l_p[st1_x:ed1_x],l_p[60000:114000]],axis=0)
                l_p_test = np.concatenate([l_p[st2_x:ed2_x],l_p[114000:120000]],axis=0)  

            print ("Loading Signal to Noise Ratio - information")
            path_SNR = "/KDD_Submission/Data/Data_X_snr_Ylabel_120.npz"
            x_snr_r = np.load(path_SNR, allow_pickle=True)['X'][1,:]
            boolArr_thresh_20 = x_snr_r > 20
            x_snr_ind = boolArr_thresh_20
            x_snr_ind_test = np.concatenate([x_snr_ind[st2_x:ed2_x],x_snr_ind[114000:120000]],axis=0)
            self.x_snr_r_test = np.concatenate([x_snr_r[st2_x:ed2_x],x_snr_r[114000:120000]],axis=0)

            path_mu = "/KDD_Submission/Data/Data_X_mu_Ylabel_120.npz"
            x_mu = np.load(path_mu, allow_pickle=True)['X'][0,:]
            self.x_mu_test = np.concatenate([x_mu[st2_x:ed2_x],x_mu[114000:120000]],axis=0)    
            return (x_train, y_train,l_p_train), (x_test, y_test,l_p_test,x_snr_ind_test)

        (x_train, y_train,l_p_train), (x_test, y_test,l_p_test,x_snr_ind_test) = load_data(loc,lab_loc,120000)

        n_p = 3
        l_p_train_radians = np.deg2rad(l_p_train[:,2])
        l_p_test_radians = np.deg2rad(l_p_test[:,2])

        e1_train =  ((1.0-l_p_train[:,1])/(1.0+l_p_train[:,1])) * np.cos(2*l_p_train_radians)
        e2_train =  ((1.0-l_p_train[:,1])/(1.0+l_p_train[:,1])) * np.sin(2*l_p_train_radians)

        e1_test =  ((1.0-l_p_test[:,1])/(1.0+l_p_test[:,1])) * np.cos(2*l_p_test_radians)
        e2_test =  ((1.0-l_p_test[:,1])/(1.0+l_p_test[:,1])) * np.sin(2*l_p_test_radians)

        l_p_train[:,1] = e1_train
        l_p_train[:,2] = e2_train

        l_p_test[:,1] = e1_test
        l_p_test[:,2] = e2_test

        l_p_train = l_p_train[:,0:n_p].reshape(-1,n_p)
        l_p_test = l_p_test[:,0:n_p].reshape(-1,n_p)

        max_lensed_inp = np.max(np.concatenate([l_p_train,l_p_test],axis=0),axis=0)
        min_lensed_inp = np.min(np.concatenate([l_p_train,l_p_test],axis=0),axis=0)

        only_lensed = True

        #train
        if not only_lensed:
            print ("loading X for all lensed+unlensed data")
            ind_slice = 1 
            strt = 0 
            end = 12000
            l_p_train = (l_p_train[0:108000]-min_lensed_inp)/(max_lensed_inp - min_lensed_inp)
            l_p_test = (l_p_test[strt:end:ind_slice]-min_lensed_inp)/(max_lensed_inp - min_lensed_inp)

            self.imgs = torch.from_numpy(x_train[0:108000]).float() 
            self.imgs_test = torch.from_numpy(x_test[strt:end:ind_slice]).float()
            self.lens_p = torch.from_numpy(l_p_train).float()
            self.lens_p_test = torch.from_numpy(l_p_test).float()
            self.label = torch.from_numpy(y_train[0:108000]).float() 
            self.label_test = torch.from_numpy(y_test[strt:end:ind_slice]).float() 

            self.x_snr_ind_test = x_snr_ind_test[strt:end:ind_slice]
            self.x_snr_r_test = self.x_snr_r_test[strt:end:ind_slice]
            self.x_mu_test = self.x_mu_test[strt:end:ind_slice]

        else:
            print ("loading X for only lensed data")

            l_p_train_lensed = (l_p_train[0:54000]-min_lensed_inp)/(max_lensed_inp - min_lensed_inp)
            l_p_test_lensed = (l_p_test[0:6000]-min_lensed_inp)/(max_lensed_inp - min_lensed_inp)

            self.imgs = torch.from_numpy(x_train[0:54000]).float() 
            self.imgs_test = torch.from_numpy(x_test[0:6000]).float()
            self.lens_p = torch.from_numpy(l_p_train_lensed).float()
            self.lens_p_test = torch.from_numpy(l_p_test_lensed).float()
            self.label = torch.from_numpy(y_train[0:54000]).float() 
            self.label_test = torch.from_numpy(y_test[0:6000]).float()     
            self.x_snr_ind_test = x_snr_ind_test[0:6000]      

    def __len__(self):
        return self.imgs.size(0)
    @property
    def ndim(self):
        return self.imgs.size(1)

    def __getitem__(self, index):
        x = self.imgs[index].view(3, 111, 111)
        y = self.lens_p[index].view(3)
        label = self.label[index].view(1)
        return x,y,label

    def __getmax__(self):
        return torch.max(self.imgs).tolist()




if __name__ == '__main__':

   StrongLensing_data = StrongLensing_new()
   print (StrongLensing_data.__len__)
   x,y,label = StrongLensing_data.__getitem__(1000)
   print (torch.max(x),torch.min(x))

