import math
import os
import torch
torch.manual_seed(0)
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable

import lib.utils as utils
import lib.datasets as dset
from metric_helpers.loader import load_model_and_dataset
import numpy as np
np.random.seed(0)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss,accuracy_score,precision_score,recall_score
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
import scipy.stats as stats

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpt', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='.')
    args = parser.parse_args()

    if args.gpu != 0:
        torch.cuda.set_device(args.gpu)
    vae, dataset, cpargs = load_model_and_dataset(args.checkpt)

    print (cpargs)

    a=b

    save_path = cpargs.save

    x_test = dataset.imgs_test    
    if (cpargs.problem == 'HEP_SL'):
        x_test = x_test.cuda()
        y_test = dataset.lens_p_test
        y_test = y_test.cuda()
        label_test = dataset.label_test
        label_test = label_test.cuda()

        test_imgs = x_test
        test_imgs = test_imgs.view(-1, 3,111,111)
        print ("test_data_size",x_test.size())

        x_snr_ind_test = dataset.x_snr_ind_test
        x_snr_r_test = dataset.x_snr_r_test 
        x_mu_test = dataset.x_mu_test

    elif (cpargs.problem == 'Climate_ORNL'):
        x_test = dataset.imgs#_test
        train_set = dset.Climate_ESM_LU_data_mlp()
        img_max = train_set.__getmax__()
        print ("img_max",img_max,x_test.shape)
        x_test = x_test.float().div(img_max)
        x_test = x_test.cuda()
        y_test = torch.Tensor(np.zeros((len(x_test),3)))
        y_test = y_test.cuda()
        label_test = torch.Tensor(np.zeros(len(x_test)))
        label_test = label_test.cuda() 
        test_imgs = x_test
        test_imgs = test_imgs.view(-1,42660)
        print ("test_data_size",x_test.size())        

    _, reco_imgs, zs, z_params = vae.reconstruct_img(test_imgs,y_test,label_test)

    if not cpargs.classification:

        A = reco_imgs.cpu().data.numpy()
        B = y_test.cpu().data.numpy()
        Z_samples_mat = np.zeros((zs.shape[0],zs.shape[1],50))
        X_samples_mat = np.zeros((zs.shape[0],3,50))
        for i in range(50):
            z_samples = vae.q_dist.sample(params=z_params.cuda())
            Z_samples_mat[:,:,i] = z_samples.detach().cpu().numpy()
            x_params = vae.decoder.forward(z_samples).view(z_samples.size(0), 3)
            xs = vae.x_dist.sample(params=x_params,latent=False)
            X_samples_mat[:,:,i] = x_params.detach().cpu().numpy()

        path_Z = "./Samples_Z.npz"
        path_X = "./Samples_X.npz"
        np.savez_compressed(path_Z, Z=Z_samples_mat)
        np.savez_compressed(path_X, X=X_samples_mat)

        Z_samples_mat = np.load(path_Z, allow_pickle=True)['Z']
        X_samples_mat = np.load(path_X, allow_pickle=True)['X']

        error_std = np.std(np.exp(X_samples_mat),axis = 2)
        std_X = np.std((X_samples_mat),axis = 2)
        mean_X = np.mean((X_samples_mat),axis = 2)

        print ("RMSE-all", np.sqrt((np.square(A - B)).mean(axis=None)))
        print ("MAE-all", np.abs(A - B).mean(axis=None))

        log_post = stats.norm.logpdf(B, loc = mean_X, scale = std_X)
        MLPD = np.mean(log_post)
        MdLPD = np.median(log_post)

        print ("(MLPD,MdLPD",MLPD,MdLPD)

        print ("RMSE-all-com", np.sqrt((np.square(A - mean_X)).mean(axis=None)))
        print ("MAE-all-com", np.abs(A - mean_X).mean(axis=None))

        def Plot_y_eq_x(ax1):
            a1,a2 = ax1.get_xlim()
            b1,b2 = ax1.get_ylim()
            c1 = np.min([a1,b1])
            c2 = np.max([a2,b2])
            ax1.set_xlim([c1,c2])
            ax1.set_ylim([c1,c2])
            ax1.plot([c1,c2],[c1,c2],'r-')

        fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(30, 30))
        for i in range(3):
            ax[i].errorbar(B[:,i],A[:,i], yerr=(error_std[:,i]),fmt='.', color='black',
                ecolor='blue', elinewidth=1, capsize=0)
            Plot_y_eq_x(ax[i])
            plt.tight_layout()
            ax[i].set_ylabel('Predicted', fontsize=20) 
            ax[i].set_xlabel('Observed', fontsize=20) 
            ax[i].tick_params(axis='both', which='major', labelsize=18)
            ax[i].set_aspect(aspect=1)
        fig.savefig(utils.save_plot_fname(cpargs.save, 'UQ_Monte Carlo.png'))



        # Scatter plot of Z colored by the regression parameters
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(zs.cpu().data.numpy())
        fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(30, 10))
        im1 = ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1],c=B[:,0],cmap=plt.cm.Spectral)
        im2 = ax[1].scatter(X_tsne[:, 0], X_tsne[:, 1],c=B[:,1],cmap=plt.cm.Spectral)
        im3 = ax[2].scatter(X_tsne[:, 0], X_tsne[:, 1],c=B[:,2],cmap=plt.cm.Spectral)
        fig.colorbar(im1, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])
        fig.colorbar(im3, ax=ax[2])
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[2].set_aspect('equal')
        ax[0].title.set_text('Einstein Radius')
        ax[1].title.set_text('Complex Ellipticity 1')
        ax[2].title.set_text('Complex Ellipticity 2')
        

        fig.savefig(utils.save_plot_fname(cpargs.save, 'Latent_space_TSNE_by_output_'+cpargs.name_save+'.png'))


    else:

        logpx = vae.x_dist.log_density(label_test, params=reco_imgs,latent=False)
        
        Entropy = vae.x_dist.Entropy(label_test, params=reco_imgs,latent=False)
        logpx_L = vae.x_dist.log_density(label_test.mul(0), params=reco_imgs,latent=False)
        logpx_UL = vae.x_dist.log_density(label_test.mul(0).add(1), params=reco_imgs,latent=False)

        #### Calculate the Rate metric R = P(Y|Z) P(Z)
        logpz = vae.prior_dist.log_density(zs, params=reco_imgs)
        Rate = logpx.exp()*logpz.exp()

        reco_imgs = reco_imgs.sigmoid()
        A = reco_imgs.cpu().data.numpy()
        B = label_test.cpu().data.numpy()
        X_TEST = test_imgs.cpu().data.numpy()

        print ("X_TEST.shape",X_TEST.shape)

        print ("A.shape",A.shape,"B.shape",B.shape)
        print ("Num of Non zeros",np.count_nonzero(A),np.count_nonzero(B))
        print ("Num of p > 0.5",len(A[A>0.5]), "Num of p <= 0.5",len(A[A<=0.5]))
        print ("max,min",np.max(A),np.max(B),np.min(A),np.min(B))

 
        ind_B_true_obs = B==1.0
        ind_B_false_obs = B==0.0


        A_true_pred = A[ind_B_true_obs]
        ind_A_true_pred_true = A_true_pred>0.5
        A_true_pred_true = A_true_pred[A_true_pred>0.5]
        ind_A_true_pred_false = A_true_pred<=0.5
        A_true_pred_false = A_true_pred[A_true_pred<=0.5]

        B_true_pred = A[ind_B_false_obs]
        ind_B_true_pred_true = B_true_pred<=0.5
        B_true_pred_true = B_true_pred[B_true_pred<=0.5]
        ind_B_true_pred_false = B_true_pred>0.5
        B_true_pred_false = B_true_pred[B_true_pred>0.5]


        print ("True Positive - obs=1(UL),pred=1(UL)  :", len(A_true_pred_true))
        print ("True Negative - obs=0(L),pred=0(L)  :", len(B_true_pred_true ))
        print ("False Positive - obs=0(L),pred=1(UL)  :", len(B_true_pred_false))
        print ("False Negative - obs=1(UL),pred=0(L)  :", len(A_true_pred_false))

        Acc_TP = len(A_true_pred_true)/(len(A_true_pred_true)+len(A_true_pred_false ))
        Acc_TN =len(B_true_pred_true)/(len(B_true_pred_true)+len(B_true_pred_false ))
        Acc_mean=(len(A_true_pred_true)+len(B_true_pred_true))/(len(A_true_pred_true)+len(B_true_pred_true)+len(A_true_pred_false)+len(B_true_pred_false))

        print ("Accuracy TPR",Acc_TP)
        print ("Accuracy TNR",Acc_TN)
        print ("mean Accuracy",Acc_mean )

        # accuracy: (tp + tn) / (p + n)
        accuracy = (len(A_true_pred_true) + len(B_true_pred_true))/ (len(A_true_pred_true) + len(B_true_pred_true)+len(B_true_pred_false)+len(A_true_pred_false))
        print (B.astype(np.int).shape)
        print('Accuracy: %f' % accuracy, 'Accuracy: %f' % accuracy_score(B.astype(np.int),np.around(A)))
        # precision tp / (tp + fp)
        precision = len(A_true_pred_true)/(len(A_true_pred_true)+len(B_true_pred_false))
        print('Precision: %f' % precision, 'Precision: %f' % precision_score(B.astype(np.int),np.around(A)))
        # recall: tp / (tp + fn)
        recall = len(A_true_pred_true)/(len(A_true_pred_true)+len(A_true_pred_false))
        print('Recall: %f' % recall,'Recall: %f' % recall_score(B.astype(np.int),np.around(A)))
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = 2*len(A_true_pred_true)/(2*len(A_true_pred_true)+len(B_true_pred_false)+len(A_true_pred_false))
        print('F1 score: %f' % f1,'F1 score: %f' % f1_score(B.astype(np.int),np.around(A)))
        
        # kappa
        kappa = cohen_kappa_score(B.astype(np.int),np.around(A))
        print('Cohens kappa: %f' % kappa)
        # ROC AUC
        auc = roc_auc_score(B.astype(np.int), logpx_UL.exp().cpu().data.numpy())
        print('ROC AUC: %f' % auc)
        # confusion matrix
        matrix = confusion_matrix(B.astype(np.int),np.around(A))
        print(matrix) 

        # Scatter plot of Z colored by the label class
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(zs.cpu().data.numpy())
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(20, 10))
        im1 = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],c=B[:,0],cmap=plt.cm.Spectral)
        fig.colorbar(im1, ax=ax)
        ax.set_aspect('equal')
        ax.title.set_text('Binary Class')
        fig.savefig(utils.save_plot_fname(save_path, Tag+'Latent_space_TSNE_by_class_'+cpargs.name_save+'.png'))
 