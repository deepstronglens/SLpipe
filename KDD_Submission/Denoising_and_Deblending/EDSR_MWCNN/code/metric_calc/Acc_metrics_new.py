import argparse
import numpy as np
from skimage import measure
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_nrmse as nmse

class Strong_Lensing_results():

    '''
    module_select : Denoise, Deblend, Classification, Regression
    data_stage : Train, Test, Inference
    Metric_calc : True / False
    '''
    def __init__(self, Data_path=None, x_snr_ind = None, x_snr_ind_true=None):

        print ("Calculating Metrics")       
        if (x_snr_ind_true is not None):
            self.snr_remove = True
        else: 
            self.snr_remove = False
        self.x_snr_ind = x_snr_ind
        self.x_snr_ind_true = x_snr_ind_true
        
        self.name_path = Data_path
        print ("self.name_path",self.name_path)
        self.inference = True
        self.indices_train_lensed = np.load('/KDD_Submission/Denoising_Deblending/indices_lensed.npy')
        self.indices_train_unlensed = np.load('/KDD_Submission/Denoising_Deblending/indices_unlensed.npy')
        self.indices_test = np.concatenate([self.indices_train_lensed[9000:],self.indices_train_unlensed[9000:]],axis=0)        
        self.Metrics_calc_npz()
        self.n_folders = (len(self.name_path))



    def Metrics_calc_npz(self):
        print ("Metrics Calculation npz")
        n_folders = (len(self.name_path))
        print ("num_folders",n_folders)
        k = 0
        #TODO: Make this automatic or expose to user

 

        if not (self.snr_remove):
            if (self.inference):
                start_mat = np.array([0,0]) 
                end_mat = np.array([6000,6000]) 
                num_im = np.sum(end_mat-start_mat)
            else:

                start_mat = np.array([0]) 
                end_mat = np.array([2000]) 
                num_im = np.sum(end_mat-start_mat)            

        else:
            if (n_folders == 2):
                start_mat = np.array([0,0]) 
                end_mat = np.array([60000,60000]) 
                self.x_snr_ind_1 = self.x_snr_ind[start_mat[0]:end_mat[0]]
                self.x_snr_ind_2 = self.x_snr_ind[end_mat[0]+start_mat[1]:end_mat[1]+end_mat[0]]
                self.x_snr_ind_true1 = np.where(self.x_snr_ind_1)[0]
                self.x_snr_ind_true2 = np.where(self.x_snr_ind_2)[0]
            num_im1 = len(self.x_snr_ind_true1)
            num_im2 = len(self.x_snr_ind_true2)
            start_mat = np.array([0,0]) 
            end_mat = np.array([num_im1,num_im2]) 
            num_im = np.sum(end_mat-start_mat)

            print ("num_im1,num_im2,num_im",num_im1,num_im2,num_im)


        print ("Expected_data_size",num_im)
        Metrics_mat_LR = np.zeros((num_im,3))
        Metrics_mat_SR = np.zeros((num_im,3))

        for i in range(n_folders):
            start = start_mat[i]
            end = end_mat[i]            
            name_LR = self.name_path[i]+'Array_LR.npz'
            name_HR = self.name_path[i]+'Array_HR.npz'
            name_SR = self.name_path[i]+'Array_SR.npz'
            with np.load(name_LR, allow_pickle=True) as f:
                X_LR = f['X']
                print ("input X shape",X_LR.shape)
                #X = X#.transpose(1,0,2,3) # use the transpose when loading from Data_XYlabel_fimg_120_new.npz
            with np.load(name_HR, allow_pickle=True) as f:
                X_HR = f['X']
                print ("input X shape",X_HR.shape) 
            with np.load(name_SR, allow_pickle=True) as f:
                X_SR = f['X']
                print ("input X shape",X_SR.shape) 

            if (self.snr_remove):
                if (i==0):
                    X_LR2 = X_LR[self.x_snr_ind_true1]
                    X_HR2 = X_HR[self.x_snr_ind_true1]
                    X_SR2 = X_SR[self.x_snr_ind_true1]
                elif (i==1):
                    X_LR2 = X_LR[self.x_snr_ind_true2]
                    X_HR2 = X_HR[self.x_snr_ind_true2]
                    X_SR2 = X_SR[self.x_snr_ind_true2]
            else:
                X_LR2 = X_LR
                X_HR2 = X_HR
                X_SR2 = X_SR             

            for j in range (start,end):

                j = j+54000
                img_LR=X_LR2[j,:,:,:].astype(np.uint8).transpose(1,2,0) 
                img_HR=X_HR2[j,:,:,:].astype(np.uint8).transpose(1,2,0) 
                img_SR=X_SR2[j,:,:,:].astype(np.uint8).transpose(1,2,0) 

                Metrics_mat_LR[k,0] = nmse(img_HR,img_LR)
                Metrics_mat_LR[k,1] = ssim(img_HR,img_LR,data_range=img_HR.max() - img_HR.min(),multichannel=True)
                Metrics_mat_LR[k,2] = psnr(img_HR,img_LR, data_range=img_HR.max() - img_HR.min()) 
                
                Metrics_mat_SR[k,0] = nmse(img_HR,img_SR)
                Metrics_mat_SR[k,1] = ssim(img_HR,img_SR, data_range=img_HR.max() - img_HR.min(),multichannel=True)  
                Metrics_mat_SR[k,2] = psnr(img_HR,img_SR, data_range=img_HR.max() - img_HR.min())

                k = k+1

        print ("Extracted_data_size",k)

        print ("Metrics:----NMSE-------SSIM-------PSNR")
        print ("LR-Mean",np.mean(Metrics_mat_LR,axis=0))
        print ("SR-Mean",np.mean(Metrics_mat_SR,axis=0))

        print ("LR-Std",np.std(Metrics_mat_LR,axis=0))
        print ("SR-Std",np.std(Metrics_mat_SR,axis=0))


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--path_SNR', type=str, default='/KDD_Submission/Denoising_Deblending/EDSR_MWCNN/Data_Regression/Data_X_snr_Ylabel_120.npz', help='Signal to Noise Ratio')
    parser.add_argument('--thresh', type=int, default=20,help='SNR threshold to use')
    parser.add_argument('--path_results', type=str, default=['/KDD_Submission/Denoising_Deblending/EDSR_MWCNN/experiment/HEP_Denoise_and_Deblend_EDSR_2Loss_Scratch_20k_V4_4_Inference_2/results/'], nargs='+', help='Path to the reslts folder - can be a list')
    #parser.add_argument('--path_results', type=str, default=['/KDD_Submission/Denoising_Deblending/EDSR_MWCNN/experiment/HEP_Denoise_and_Deblend_EDSR_2Loss_Scratch_20k_V4_4/results/'], nargs='+', help='Path to the reslts folder - can be a list')
    parser.add_argument('--ind_start', type=int, default=0, help='Starting index of array to use - results')
    parser.add_argument('--ind_end', type=int, default=120000, help='Last index of array to use - results')

    args = parser.parse_args() 
    path_SNR = args.path_SNR 
    x_snr_r = np.load(path_SNR, allow_pickle=True)['X'][1,:]
    boolArr_thresh_20 = x_snr_r > args.thresh

    x_snr_ind = boolArr_thresh_20[args.ind_start:args.ind_end]

    x_snr_ind_true = np.where(x_snr_ind)[0]   

    Data_path = args.path_results

    Results = Strong_Lensing_results(Data_path=Data_path,x_snr_ind = x_snr_ind,x_snr_ind_true=None)#x_snr_ind_true)