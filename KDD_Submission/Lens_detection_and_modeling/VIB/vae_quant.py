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
import Vizualization as Viz_plot


class ConvEncoder_HEP_SL(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_HEP_SL, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)  # 30 x 20
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 4, 2, 1)  # 15 x 10
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, 3, 1)  # 7 x 5
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 4, 3, 1)  # 3 x 2
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3)    # 1 X 1
        self.bn5 = nn.BatchNorm2d(256) 
        self.conv_z = nn.Conv2d(256, output_dim, (1,1)) #1 X 1

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 3,111, 111)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z

class ConvDecoder_HEP_SL(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_HEP_SL, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 256, (1,1), 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(256, 128, 3)  # 3 X 2
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 128, 5,2,0)  # 7 X 5
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 64, 5, 3, 1)  # 15 X 10
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(64, 64, 5, 2, 1)  # 30 X 20
        self.bn5 = nn.BatchNorm2d(64)
        self.conv_final = nn.ConvTranspose2d(64, 3, 5,2,1) #61 X 41

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)
        self.act_end = nn.Sigmoid()
    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img

class MLPDecoder_y(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(MLPDecoder_y, self).__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 10),
            nn.Tanh(),
            nn.Linear(10, out_dim)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), self.out_dim)
        return mu_img


class MLPDecoder_label(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder_label, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1)
        return mu_img


class VAE(nn.Module):
    def __init__(self, z_dim, beta, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 x_dist =dist.Bernoulli(), x_dist_name='bernoulli', include_mutinfo=True, tcvae=False, 
                 conv=False, mss=False, problem='HEP_SL', VIB=False, UQ=False,classification=False):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = beta 
        self.mss = mss
        self.x_dist = x_dist 
        self.VIB = VIB 
        self.conv = conv
        self.x_dist_name = x_dist_name
        self.problem = problem
        self.UQ = UQ
        self.classification = classification

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if conv:
            self.encoder = ConvEncoder_HEP_SL(z_dim * self.q_dist.nparams)

            if (self.VIB):
                if not self.classification:
                    if (self.UQ):
                        out_dim = 6
                    else:
                        if (self.problem == 'HEP_SL'):
                            out_dim = 3 
                    self.decoder = MLPDecoder_y(z_dim,out_dim)
                else:
                    self.decoder = MLPDecoder_label(z_dim) # binary classification
            else:
                if (self.problem == 'HEP_SL'):
                    self.decoder = ConvDecoder_HEP_SL(z_dim)
                
        else:
            print ("not implemented")              

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x): 
        if (self.problem == 'HEP_SL'):
            x = x.view(x.size(0), 3, 111, 111)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x)
        z_params = z_params.view(x.size(0), self.z_dim, self.q_dist.nparams)
        
        #sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        if (self.VIB):
            if not self.classification:
                if (self.UQ):
                    x_params = self.decoder.forward(z).view(z.size(0), 6)
                    x_params = x_params.reshape(z.size(0), 3, 2)
                    latent = True
                else:
                    if (self.problem == 'Climate_ORNL'):
                        x_params = self.decoder.forward(z).view(z.size(0), 64)
                    elif (self.problem == 'HEP_SL'):
                        x_params = self.decoder.forward(z).view(z.size(0), 3)
                    elif (self.problem == 'Nuclear_Physics'):
                        x_params = self.decoder.forward(z).view(z.size(0),200)
                    latent = False 
            else:
                x_params = self.decoder.forward(z).view(z.size(0), 1)
                latent = False                               
        else:
            latent=False
            if (self.problem == 'Climate_ORNL'):
                x_params = self.decoder.forward(z).view(z.size(0), 42660) 
            elif (self.problem == 'HEP_SL'):
                x_params = self.decoder.forward(z).view(z.size(0), 3, 111, 111)
            elif (self.problem == 'Nuclear_Physics'):
                print ("Not Implemented")


        if (self.x_dist_name == 'bernoulli'):
           x_params = x_params

        xs = self.x_dist.sample(params=x_params,latent=latent)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x, y,label):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo(self, x, y, label,dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        if (self.problem == 'HEP_SL'):
            x = x.view(batch_size, 3, 111, 111)

        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x,y,label)
        if (self.problem == 'HEP_SL'):
            if (self.VIB):
                if not self.classification:
                    if (self.UQ):
                        x_params = x_params.reshape(batch_size, 3, 2)
                        x_params = x_params.view(batch_size, 3, 2)
                    else: 
                        x_params = x_params.view(batch_size, 3)
                else:
                    x_params = x_params.view(batch_size, 1)
            else:
                x_params = x_params.view(batch_size, 3, 111,111)

        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        if (self.VIB):
            if not self.classification:
                if (self.UQ):
                    latent = True
                else:
                    latent = False
                logpx = self.x_dist.log_density(y, params=x_params,latent=latent).view(batch_size, -1).sum(1)
                _logpxz = self.x_dist.log_density(y.view(batch_size, 1, 64),
                        x_params.view(1,batch_size, 64),latent=latent)
            else:
                latent = False
                logpx = self.x_dist.log_density(label, params=x_params,latent=latent).view(batch_size, -1).sum(1)

        else:
            if self.conv:
                latent = False
                logpx = self.x_dist.log_density(x, params=x_params,latent=latent).view(batch_size, -1).sum(1)
            else: 
                logpx = self.x_dist.log_density(x, params=x_params,latent=False).view(batch_size, -1).sum(1)
        
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)       

        elbo = logpx + logpz - logqz_condx
        if utils.isnan(elbo).any():
           if utils.isnan(logpx).any():
              print ("NAN in logpx")
           elif utils.isnan(logpz).any():
              print ("NAN in logpz")
           elif utils.isnan(logqz_condx).any():
              print ("NAN in logqz_condx")
              if utils.isnan(zs).any():
                  print ("NAN in zs")
              elif utils.isnan(z_params).any():
                  print ("NAN in z_params")

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach(),x_params,z_params

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        # compute log q(h_f) ~= log 1/(NM) sum_m=1^M q(h_f|zx_m) = - log(MN) + logsumexp_m(q(h|zx_m))
        # h_f is the output of VIB decoder, zx_m is the latent dimension of the VIB
        
        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))

            if not self.classification:
                if (self.VIB):
                    _logqx = (logsumexp(_logpxz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
                else:
                    _logqx = torch.Tensor(np.zeros(len(logqz)))
            else:
                _logqx = torch.Tensor(np.zeros(len(logqz)))  
               
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

            if not self.classification:
                if (self.VIB):
                    logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logpxz.data))
                    _logqx = logsumexp(logiw_matrix + _logpxz.sum(2), dim=1, keepdim=False)
                else:
                    _logqx = torch.Tensor(np.zeros(len(logqz)))            
            else:
                _logqx = torch.Tensor(np.zeros(len(logqz)))  

        if not self.tcvae:
            if self.include_mutinfo:
                #print ("no_tcvae + mutualinfo")
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                #print ("no_tcvae + no_mutualinfo")
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                #print ("tcvae + mutualinfo")
                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                #print ("tcvae + no_mutualinfo")
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo.detach(),x_params,z_params,_logqx, logqz


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
    if args.problem == 'shapes':
        train_set = dset.Shapes()
    elif args.problem == 'faces':
        train_set = dset.Faces()
    elif args.problem == 'HEP_SL':
        train_set = dset.StrongLensing_new()
    else:
        raise ValueError('Unknown dataset ' + str(args.problem))

    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def anneal_kl(args, vae, iteration):
    if args.problem == 'shapes':
        warmup_iter = 7000
    elif args.problem == 'faces':
        warmup_iter = 2500

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-d', '--dataset', default='faces', type=str, help='dataset name',
        choices=['shapes', 'faces'])
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-x_dist', default='normal', type=str, choices=['normal', 'bernoulli'])
    parser.add_argument('-n', '--num-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=2048, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-z', '--latent-dim', default=10, type=int, help='size of latent dimension')
    parser.add_argument('--beta', default=1, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_false')
    parser.add_argument('--beta-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
    parser.add_argument('--save', default='test2')
    parser.add_argument('--log_freq', default=50, type=int, help='num iterations per log')
    parser.add_argument('-problem', default='Climate_ORNL', type=str, choices=['HEP_SL', 'Climate_ORNL', 'Climate_C','Nuclear_Physics'])
    parser.add_argument('--VIB', action='store_true', help='VIB regression')
    parser.add_argument('--UQ', action='store_true', help='Uncertainty Quantification - likelihood')
    parser.add_argument('-name_S', '--name_save', default=[], type=str, help='name to save file')
    parser.add_argument('--classification', action='store_true')
    parser.add_argument('--Func_reg', action='store_true')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    # data loader
    train_loader = setup_data_loaders(args, use_cuda=True)

    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

    # setup the likelihood distribution
    if args.x_dist == 'normal':
        x_dist = dist.Normal() 
    elif args.x_dist == 'bernoulli':   
        x_dist = dist.Bernoulli()
    else:
        raise ValueError('x_dist can be Normal or Bernoulli')

    

    vae = VAE(z_dim=args.latent_dim, beta=args.beta, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, 
           x_dist=x_dist, x_dist_name=args.x_dist, include_mutinfo=not args.exclude_mutinfo, 
           tcvae=args.tcvae, conv=args.conv, mss=args.mss, problem = args.problem, VIB=args.VIB, UQ=args.UQ, classification=args.classification)

    if (args.Func_reg):
        args.latent_dim2 =  4
        args.beta2 = 0.0
        prior_dist2 = dist.Normal()
        q_dist2 = dist.Normal()
        x_dist2 = dist.Normal()
        args.x_dist2 = dist.Normal()
        args.tcvae2 = False
        args.conv2 = False
        args.problem2 = 'Climate_ORNL'
        args.VIB2 = True
        args.UQ2 = False
        args.classification2 = False

        vae2 = VAE(z_dim=args.latent_dim2, beta=args.beta2, use_cuda=True, prior_dist=prior_dist2, q_dist=q_dist2, 
            x_dist=x_dist2, x_dist_name=args.x_dist2, include_mutinfo=not args.exclude_mutinfo, 
            tcvae=args.tcvae2, conv=args.conv2, mss=args.mss, problem = args.problem2, VIB=args.VIB2, UQ=args.UQ2, classification=args.classification2)

    # setup the optimizer
    #optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
    if (args.Func_reg):
        params = list(vae.parameters()) + list(vae2.parameters())
        optimizer = optim.RMSprop(params, lr=args.learning_rate)
    else:
        optimizer = optim.RMSprop(vae.parameters(), lr=args.learning_rate)
    # setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(env=args.save, port=4500)

    train_elbo = []
    train_rmse = []
    train_mae = []
    train_elbo1 = []
    train_elbo2 = []
    train_elbo3 = []
    train_elbo4 = []    
    train_rmse2 = []
    train_mae2 = []
    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    print ("num_iteration",len(train_loader),args.num_epochs)
    iteration = 0
    print ("likelihood function",args.x_dist,x_dist)
    
    train_iter = iter(train_loader)
    images = train_iter.next()

    img_max = train_loader.dataset.__getmax__()

    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    elbo_running_rmse = utils.RunningAverageMeter()
    elbo_running_mae = utils.RunningAverageMeter()
    elbo_running_mean1 = utils.RunningAverageMeter()
    elbo_running_mean2 = utils.RunningAverageMeter()
    elbo_running_mean3 = utils.RunningAverageMeter()
    elbo_running_mean4 = utils.RunningAverageMeter()
    elbo_running_rmse2 = utils.RunningAverageMeter()
    elbo_running_mae2 = utils.RunningAverageMeter()
    #plot the data to visualize

    x_test = train_loader.dataset.imgs_test
    x_train = train_loader.dataset.imgs

    def count_parameters(model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return (trainable,total)

    while iteration < num_iterations:
        for i, xy in enumerate(train_loader):
            iteration += 1
            batch_time = time.time()
            vae.train()
            #anneal_kl(args, vae, iteration)
            optimizer.zero_grad()
            # transfer to GPU      
            if (args.problem == 'HEP_SL'):
                x = xy[0]              
                x = x.float()
                x = x.cuda()
                x = Variable(x)

                y = xy[1] 
                y = y.cuda()
                y = Variable(y)

                label = xy[2]
                label = label.cuda()
                label = Variable(label)

            # Get the Training Objective
            obj, elbo, x_mean_pred, z_params1,_,_ = vae.elbo(x,y,label,dataset_size)
            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
            
            obj.mean().mul(-1).backward()
            elbo_running_mean.update(elbo.mean().data)#[0])
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                
                if (args.VIB):
                    if not args.classification:
                        if (args.UQ):
                            A = x_mean_pred.cpu().data.numpy()[:,:,0]
                        else:
                            A = x_mean_pred.cpu().data.numpy()
                        B = y.cpu().data.numpy()
                    else:
                        A = x_mean_pred.cpu().data.numpy() 
                        B = label.cpu().data.numpy()
                else:
                    A = x_mean_pred.cpu().data.numpy()
                    B = x.cpu().data.numpy()
                
                
                rmse = np.sqrt((np.square(A - B)).mean(axis=None))
                mae = np.abs(A - B).mean(axis=None)

                elbo_running_rmse.update(rmse)
                elbo_running_mae.update(mae)

                train_rmse.append(elbo_running_rmse.avg)
                train_mae.append(elbo_running_mae.avg)

                print('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f) RMSE: %.4f (%.4f) MAE: %.4f (%.4f)' % (
                    iteration, time.time() - batch_time, vae.beta, vae.lamb,
                    elbo_running_mean.val, elbo_running_mean.avg,elbo_running_rmse.val, elbo_running_rmse.avg,
                    elbo_running_mae.val, elbo_running_mae.avg))

                utils.save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args}, args.save, 0)


                print ("max pred:",np.max(A), "max input:",np.max(B), "min pred:",np.min(A), "min input:",np.min(B))

    if (args.problem == 'HEP_SL'):
        x_test = train_loader.dataset.imgs_test
        x_test = x_test.cuda()
        y_test = train_loader.dataset.lens_p_test
        y_test = y_test.cuda()
        label_test = train_loader.dataset.label_test
        label_test = label_test.cuda() 

    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.save, 0)
    name_save = args.name_save
    

    Viz_plot.Convergence_plot(train_elbo,train_rmse,train_mae,name_save,args.save)
    Viz_plot.display_samples_pred_mlp(vae, x_test,y_test,label_test,args.problem,args.VIB,name_save,args.UQ,args.classification,args.save,img_max)
    
    # Report statistics after training
    vae.eval()
    return vae


if __name__ == '__main__':

    run_time = time.time()
    model = main()

    print ('Total time: %.2f' % (time.time()-run_time))