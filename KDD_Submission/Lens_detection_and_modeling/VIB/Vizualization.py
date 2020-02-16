import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow

from elbo_decomposition import elbo_decomposition
import numpy as np
import torchvision.transforms.functional as TF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn import manifold
from PIL import Image
import cv2
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error


def display_samples_pred_mlp(model, x, y, label, problem, VIB, name, UQ, classification,save_path,img_max):

    n_p =3

    def Plot_y_eq_x(ax1):
        a1,a2 = ax1.get_xlim()
        b1,b2 = ax1.get_ylim()
        c1 = np.min([a1,b1])
        c2 = np.max([a2,b2])
        ax1.set_xlim([c1,c2])
        ax1.set_ylim([c1,c2])
        ax1.plot([c1,c2],[c1,c2],'r-')
    
    if (problem == 'Climate_ORNL'):
        test_imgs = x.float().div(img_max)
        test_imgs = test_imgs.view(-1, 42660)

    elif (problem == 'Nuclear_Physics'):
        test_imgs = x.float()#.div(img_max)
        test_imgs = test_imgs.view(-1, 502)
        
    elif (problem == 'HEP_SL'):
        test_imgs = x
        test_imgs = test_imgs.view(-1, 3,111,111)
        print ("test_data_size",x.size())
        

    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs,y,label)

    
    if (VIB):
        if not classification:
            if (UQ):
                A = reco_imgs.cpu().data.numpy()[:,:,0]
                error = reco_imgs.cpu().data.numpy()[:,:,1]
            else:
                A = reco_imgs.cpu().data.numpy()
            B = y.cpu().data.numpy()
        else:
            A = reco_imgs.cpu().data.numpy()
            B = label.cpu().data.numpy()       
    else:
        A = reco_imgs.cpu().data.numpy()
        B = test_imgs.cpu().data.numpy()


    print ("RMSE-all", np.sqrt((np.square(A - B)).mean(axis=None)))
    print ("MAE-all", np.abs(A - B).mean(axis=None))

    if (VIB):
        if not classification:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6, 6])    
            if (UQ):
                ax.errorbar(B.flatten(),A.flatten(), yerr=np.exp(error.flatten()),linestyle='',marker='.')
            ax.plot(B.flatten(),A.flatten(), 'r.')
            plt.savefig('display_samples_pred_mlp_uq'+name+'.png')

            if  (problem == 'HEP_SL'):
                fig, ax = plt.subplots(nrows=2, ncols=n_p,figsize=(20, 20))
                for i in range(n_p):
                    ax[0,i].scatter(B[:,i],A[:,i],marker='X', alpha=0.5,color='k',s=5)
                    Plot_y_eq_x(ax[0,i])
                    if (UQ):
                        ax[1,i].errorbar(B[:,i],A[:,i], yerr=np.exp(error[:,i]),linestyle='',marker='')
                    ax[1,i].scatter(B[:,i],A[:,i],marker='.', alpha=0.5,color='r',s=5)
                    
                    Plot_y_eq_x(ax[1,i])
                    ax[0,i].set_ylabel('Predicted', fontsize=20) 
                    ax[0,i].set_xlabel('Observed', fontsize=20) 
                    ax[0,i].tick_params(axis='both', which='major', labelsize=18)
                    ax[0,i].set_aspect(aspect=1)

                    ax[1,i].set_ylabel('Predicted', fontsize=20) 
                    ax[1,i].set_xlabel('Observed', fontsize=20) 
                    ax[1,i].tick_params(axis='both', which='major', labelsize=18)
                    ax[1,i].set_aspect(aspect=1)
                    plt.tight_layout()
                fig.savefig('train_test_norm_'+name+'.png')

        else:
            print ("A.shape",A.shape,"B.shape",B.shape)
            print ("Num of Non zeros",np.count_nonzero(A),np.count_nonzero(B))
            print ("Num of p > 0.5",len(A[A>0.5]), "Num of p <= 0.5",len(A[A<=0.5]))
            print ("max,min",np.max(A),np.max(B),np.min(A),np.min(B))

            ind_B_true_obs = B==1.0
            ind_B_false_obs = B==0.0


            A_true_pred = A[ind_B_true_obs]
            A_true_pred_true = A_true_pred[A_true_pred>0.5]
            A_true_pred_false = A_true_pred[A_true_pred<=0.5]

            B_true_pred = A[ind_B_false_obs]
            B_true_pred_true = B_true_pred[B_true_pred<=0.5]
            B_true_pred_false = B_true_pred[B_true_pred>0.5]

            print ("True Positive - obs=1,pred=1  :", len(A_true_pred_true))
            print ("True Negative - obs=0,pred=0  :", len(B_true_pred_true ))
            print ("False Positive - obs=0,pred=1  :", len(B_true_pred_false))
            print ("False Negative - obs=1,pred=0  :", len(A_true_pred_false))

            Acc_TP = len(A_true_pred_true)/(len(A_true_pred_true)+len(A_true_pred_false ))
            Acc_TN =len(B_true_pred_true)/(len(B_true_pred_true)+len(B_true_pred_false ))
            Acc_mean=(len(A_true_pred_true)+len(B_true_pred_true))/(len(A_true_pred_true)+len(B_true_pred_true)+len(A_true_pred_false)+len(B_true_pred_false))

            print ("Accuracy TPR",Acc_TP)
            print ("Accuracy TNR",Acc_TN)
            print ("mean Accuracy",Acc_mean )

            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(20, 20))
            ax.scatter(B,A[:,0],marker='X', alpha=0.5,color='k',s=5)
            #Plot_y_eq_x(ax)
            ax.plot([-0.1,1.1],[-0.1,1.1],'r-')
            ax.set_ylabel('Predicted', fontsize=20) 
            ax.set_xlabel('Observed', fontsize=20) 
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.set_aspect(aspect=1)
            plt.tight_layout()
            name_save = utils.save_plot_fname(save_path, 'train_test_norm_'+name+'.png')
            fig.savefig(name_save)


    else:
        if (problem == 'HEP_SL'):
            fig, ax = plt.subplots(nrows=4, ncols=3, figsize=[30, 30])

            for i in range(4):
                img_true = test_imgs.cpu().data.numpy()[10+i].astype(np.uint8).transpose(1,2,0)
                img_pred = reco_imgs.cpu().data.numpy()[10+i].astype(np.uint8).transpose(1,2,0)
                img_pred = img_pred.clip(0, 255)
                difference = cv2.subtract(img_true, img_pred)
                difference = difference.clip(0, 255)
                print ("difference.shape,difference.max(),difference.min()",difference.shape,difference.max(),difference.min())
                ax[i,0].imshow(img_true)
                ax[i,1].imshow(img_pred)
                img = ax[i,2].imshow(difference,cmap='hsv')
                ax[i,0].set_aspect('equal')
                ax[i,1].set_aspect('equal')
                ax[i,2].set_aspect('equal')
                ax[i,0].title.set_text('Observed')
                ax[i,1].title.set_text('Predicted')
                ax[i,2].title.set_text('Difference')
                fig.colorbar(img, ax=ax[i,2])
            fig.savefig('train_test_norm_'+name+'.png')
        else:
            # Climate
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[30, 30])

            for i in range(4):
                img_true = test_imgs.cpu().data.numpy()[i,:]
                img_pred = reco_imgs.cpu().data.numpy()[i,:]
                difference = img_true - img_pred
                ax.plot(difference)
                ax.set_xlabel('Y-dimension', fontsize=20)
                ax.set_ylabel('Obs-pred', fontsize=20)           
            name_save = utils.save_plot_fname(save_path, 'train_test_norm_'+name+'.png')
            fig.savefig(name_save)



    # visualize the latent space

    # Scatter with images instead of points
    def imscatter(x, y, ax, imageData, zoom):
        images = []
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            img = imageData[i].astype(np.uint8).transpose(1,2,0)
            image = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
            images.append(ax.add_artist(ab))
        
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
    
    if (problem == 'HEP_SL'):
        fig, ax = plt.subplots()
        img = test_imgs.cpu().data.numpy()[0].astype(np.uint8).reshape([111,111,3])
        ax.imshow(img)
        fig.savefig('Visualize_X_'+name+'.png')

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(zs.cpu().data.numpy())
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots(figsize=(20, 20))
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=test_imgs.cpu().data.numpy(), ax=ax, zoom=0.2)
        fig.savefig('Latent_space_TSNE_'+name+'.png')

        # Scatter plot of Z colored by the regression parameters

        fig, ax = plt.subplots(nrows=1, ncols=n_p,figsize=(30, 10))
        im1 = ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1],c=y.cpu().data.numpy()[:,0],cmap=plt.cm.Spectral)
        im2 = ax[1].scatter(X_tsne[:, 0], X_tsne[:, 1],c=y.cpu().data.numpy()[:,1],cmap=plt.cm.Spectral)
        im3 = ax[2].scatter(X_tsne[:, 0], X_tsne[:, 1],c=y.cpu().data.numpy()[:,2],cmap=plt.cm.Spectral)
        fig.colorbar(im1, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])
        fig.colorbar(im3, ax=ax[2])
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[2].set_aspect('equal')
        ax[0].title.set_text('Einstein Radius')
        ax[1].title.set_text('Complex Ellipticity 1')
        ax[2].title.set_text('Complex Ellipticity 2')
        fig.savefig('Latent_space_TSNE_by_output_'+name+'.png')

def Convergence_plot(train_elbo,train_rmse,train_mae,name,save_path, ax=None,lntp='b.-'):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[18, 6])
    else:
        ax=ax
    ax = ax.flatten()
    ax[0].plot(train_elbo,lntp)
    ax[1].plot(train_rmse,lntp)
    ax[2].plot(train_mae,lntp)
    name_save = utils.save_plot_fname(save_path, 'Convergence_plot'+name+'.png')
    plt.savefig(name_save)

