import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import h5py

class Trainer():
    def __init__(self, args, loader, my_model1, my_loss1,ckp, my_model2=None, my_loss2=None, ckp2=None):
        self.args = args
        self.scale = args.scale
        self.use_two_opt = False
        self.ckp = ckp
        if (self.args.nmodels == 2):
            self.ckp2 = ckp2
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        #print ("Train dataset",len(self.loader_train.dataset))
        #print ("Test dataset",len(self.loader_test.dataset))
        self.model1 = my_model1
        self.loss1 = my_loss1

        self.it_ckp = 0
        if not self.args.test_only:
            n_test_data = len(self.loader_test.dataset)
            dir_save = '/gpfs/jlse-fs0/users/sand33p/stronglensing/Image_Enhancement/EDSR_MWCNN/experiment/'+self.args.save
            self. HDF5_file_model1_loc = dir_save+'/Test_checkpoint_model1.h5'
            HDF5_file_model1 = h5py.File(self. HDF5_file_model1_loc, 'a')          
            HDF5_file_model1.create_dataset("Array_HR",  (50, n_test_data, 3, 111, 111), dtype = np.float64)
            HDF5_file_model1.create_dataset("Array_LR",  (50, n_test_data, 3, 111, 111), dtype = np.float64)
            HDF5_file_model1.create_dataset("Array_SR",  (50, n_test_data, 3, 111, 111), dtype = np.float64)
            HDF5_file_model1.create_dataset("Array_Limg",  (50, n_test_data, 3, 111, 111), dtype = np.float64)
            HDF5_file_model1.close()

        #if (self.use_two_opt == False):    
        if (self.args.nmodels == 1):
            self.optimizer1 = utility.make_optimizer(args, self.model1)
            self.scheduler1 = utility.make_scheduler(args, self.optimizer1)


        elif (self.args.nmodels == 2):
            self.model2 = my_model2    
            self.loss2 = my_loss2
            if not self.args.test_only:
                n_test_data = len(self.loader_test.dataset)
                dir_save = '/gpfs/jlse-fs0/users/sand33p/stronglensing/Image_Enhancement/EDSR_MWCNN/experiment/'+self.args.save
                self. HDF5_file_model2_loc = dir_save+'/Test_checkpoint_model2.h5'
                HDF5_file_model2 = h5py.File(self. HDF5_file_model2_loc, 'a')
                HDF5_file_model2.create_dataset("Array_HR",  (50, n_test_data, 3, 111, 111), dtype = np.float64)
                HDF5_file_model2.create_dataset("Array_LR",  (50, n_test_data, 3, 111, 111), dtype = np.float64)
                HDF5_file_model2.create_dataset("Array_SR",  (50, n_test_data, 3, 111, 111), dtype = np.float64)
                HDF5_file_model2.close()

            if (self.use_two_opt):            
                self.optimizer2 = utility.make_optimizer(args, self.model2)
                self.scheduler2 = utility.make_scheduler(args, self.optimizer2)
            else:
                self.optimizer1 = utility.make_optimizer_2models(args, self.model1, self.model2)
                self.scheduler1 = utility.make_scheduler(args, self.optimizer1)

        if self.args.load != '.':
            self.optimizer1.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'model1'+'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler1.step()
            if ((self.args.nmodels == 2) & (self.use_two_opt)):
                self.optimizer2.load_state_dict(
                    torch.load(os.path.join(ckp.dir, 'model2'+'optimizer.pt'))
                )
                for _ in range(len(ckp2.log)): self.scheduler2.step()                

        self.error_last = 1e8

    def train(self):

        self.scheduler1.step()
        self.loss1.step()
        epoch1 = self.scheduler1.last_epoch + 1
        lr = self.scheduler1.get_lr()[0]

        #self.ckp.write_log(
        #    '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch1, Decimal(lr))
        #)
        self.loss1.start_log()
        self.model1.train()

        if ((self.args.nmodels == 2) & (self.args.numloss == 2)):
            if(self.use_two_opt):
                self.scheduler2.step()
            self.loss2.step()
            if(self.use_two_opt):
                epoch2 = self.scheduler2.last_epoch + 1
                lr = self.scheduler2.get_lr()[0]

            #self.ckp2.write_log(
            #    '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch2, Decimal(lr))
            #)
            self.loss2.start_log()
            self.model2.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        list1 = list(enumerate(self.loader_train))
        #print ((list1[0][1]))

        #######
        ## for the denoising/deblending pipeline, 
        ## lr: noisy blended (sim)      >>>>> from loader_train
        ## hr: denoised blended (sim)   >>>>> from loader_train
        ## sr: denoised blended (pred)
        ## lr2: hr 
        ## hr2: denoised deblended (sim) >>>>> from loader_train
        ## sr2: denoised deblended (pred)
        ### loss = f(hr,sr) + f(hr2,sr2)
        for batch, (lr, hr, lr2, hr2, _, idx_scale) in enumerate(self.loader_train): 
            #print ("batch",batch)#,lr,hr,idx_scale)
            #print ("filename?", f)
            lr, hr = self.prepare([lr, hr])
            
            
            #lr = lr[:,0:1,:,:]
            #hr = hr[:,0:1,:,:]
            timer_data.hold()
            timer_model.tic()
            self.optimizer1.zero_grad()

            if ((self.args.nmodels == 2) & (self.use_two_opt)):
                self.optimizer2.zero_grad()

            sr = self.model1(lr, self.args.scale[idx_scale])

            #print ("Data Type of output sr1",sr.type())
            if (self.args.nmodels == 2):
                lr2 , hr2 = self.prepare([lr2, hr2])
                #lr2 , hr2 = self.prepare([sr, hr2])
                #print ("Data Type of output sr1",lr2.type())
                #hr2 = self.prepare([hr2])

                ### USE the SR output from model 1 as the LR output to the model2
                lr2 = sr


                sr2 = self.model2(lr2, self.args.scale[idx_scale])
            # print ("TRAINING-Data shape-LR",lr.size())
            # print ("TRAINING-Data shape-HR",hr.size())
            # print ("TRAINING-Data shape-SR",sr.size())

            if ((self.args.nmodels == 2) & (self.args.numloss == 1)):
                loss1 = self.loss1(sr2, hr2)
            else:
                loss1 = self.loss1(sr, hr)

            if ((self.args.nmodels == 2) & (self.args.numloss == 2)):
                loss2 = self.loss2(sr2, hr2)
                loss_comb = loss2+loss1
            
            if ((self.args.nmodels == 2) & (self.args.numloss == 2)):
                if loss_comb.item() < self.args.skip_threshold * self.error_last:
                    loss_comb.backward()
                    self.optimizer1.step()
                    if(self.use_two_opt):
                        print ("self.use_two_opt")
                        self.optimizer2.step()
                else:
                    print('Skip this batch {}! (Loss: {})'.format(
                        #batch + 1, loss.data[0]
                        batch + 1,loss2.item()
                    ))
            else:
                if loss1.item() < self.args.skip_threshold * self.error_last:
                    loss1.backward()
                    self.optimizer1.step()
                else:
                    print('Skip this batch {}! (Loss: {})'.format(
                        #batch + 1, loss.data[0]
                        batch + 1,loss1.item()
                    ))                

            timer_model.hold()

            #loss = loss1+loss2
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('\nWriting Training loss at batch{}'.format(batch+1))
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss1.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                   (batch + 1) * self.args.batch_size,
                   len(self.loader_train.dataset),
                   self.loss1.display_loss_PSNR(batch),
                   timer_model.release(),
                   timer_data.release()))

                if ((self.args.nmodels == 2) & (self.args.numloss == 2)):
                    self.ckp2.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss2.display_loss(batch),
                        timer_model.release(),
                        timer_data.release()))

                    self.ckp2.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss2.display_loss_PSNR(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        if ((self.args.nmodels == 2) & (self.args.numloss == 2)): 
            self.loss2.end_log(len(self.loader_train))
            self.error_last = self.loss2.log[-1, -1]
        else:
            self.loss1.end_log(len(self.loader_train))
            self.error_last = self.loss1.log[-1, -1]

    def test(self):
        epoch = self.scheduler1.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model1.eval()
        if (self.args.nmodels == 2):
            self.ckp2.write_log('\nEvaluation:')
            self.ckp2.add_log(torch.zeros(1, len(self.scale)))
            self.model2.eval()
           

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                if (self.args.nmodels == 2):
                    eval_acc2 = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                n_test_data = len(self.loader_test.dataset)
                print ("n_test_data",n_test_data)
                Test_pred_mat_HR = np.zeros((n_test_data,3,111,111))
                Test_pred_mat_LR = np.zeros((n_test_data,3,111,111))
                Test_pred_mat_SR = np.zeros((n_test_data,3,111,111))
                Test_pred_mat_Limg = np.zeros((n_test_data,3,111,111))

                if (self.args.nmodels == 2):
                    Test_pred_mat_HR_2 = np.zeros((n_test_data,3,111,111))
                    Test_pred_mat_LR_2 = np.zeros((n_test_data,3,111,111))
                    Test_pred_mat_SR_2 = np.zeros((n_test_data,3,111,111))
                    #Test_pred_mat_Limg_2 = np.zeros((n_test_data,3,111,111))

                k = 0
                for idx_img, (lr, hr,lr2, hr2, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    #no_eval = (hr.item() == -1)
                    no_eval = False
                    # print ("before prep", lr.dtype,hr.dtype)
                    # print ("before prep", lr.max(),lr.min(),hr.max(),hr.min())
                    if not no_eval:
                        if (self.args.model == 'MWCNN'):
                            lr, hr = self.prepare_test([lr, hr])
                        else:
                            lr, hr = self.prepare([lr, hr])

                    else:
                        if (self.args.model == 'MWCNN'):
                            lr = self.prepare_test([lr])[0]
                        else:
                            lr = self.prepare([lr])[0]

                    # print ("After prep", lr.dtype,hr.dtype)
                    # print ("After prep", lr.max(),lr.min(),hr.max(),hr.min())

                    sr1 = self.model1(lr, idx_scale)
                    if (self.args.model == 'MWCNN'):
                        hr = hr[:,:,1:,1:]
                        lr = lr[:,:,1:,1:]
                        

                    if (self.args.nmodels == 2):

                        if (self.args.Test_feed_model1_out):
                            ###### the SR from Model 1 id used as LR for model 2 
                            if not no_eval:
                                #lr2, hr2 = self.prepare([sr1, hr2])
                                #lr2, hr2 = self.prepare([lr2, hr2])
                                lr2 = sr1.to(torch.device('cuda'))
                                lr2 = utility.quantize(lr2, self.args.rgb_range)
                                hr2 = hr2.to(torch.device('cuda'))
                            else:
                                if (self.args.model == 'MWCNN'):
                                    lr2 = self.prepare_test([sr1])[0]
                                else:
                                    lr2 = self.prepare([lr2])[0]
                        
                        else:
                            if not no_eval:
                                if (self.args.model == 'MWCNN'):
                                    lr2, hr2 = self.prepare_test([lr2, hr2])
                                else:
                                    lr2, hr2 = self.prepare([lr2, hr2])

                            else:
                                if (self.args.model == 'MWCNN'):
                                    lr2 = self.prepare_test([lr2])[0]
                                else:
                                    lr2 = self.prepare([lr2])[0]                         

                        #lr2 = sr1
                        #sr1_prep = self.prepare([sr1])[0]
                        sr2 = self.model2(lr2, idx_scale)
                        if (self.args.model == 'MWCNN'):
                            #hr2 = hr[:,:,1:,1:]
                            lr2 = lr2[:,:,1:,1:]
                            sr2 = sr2[:,:,1:,1:]

                        # print ("After eval", sr.dtype)
                        # print ("After eval", sr.max(),sr.min())
                    if (self.args.model == 'MWCNN'):
                        sr1 = sr1[:,:,1:,1:]
                    sr1 = utility.quantize(sr1, self.args.rgb_range)

                    Test_pred_mat_HR[k*250:(k+1)*250,:,:,:] = hr
                    Test_pred_mat_LR[k*250:(k+1)*250,:,:,:] = lr
                    Test_pred_mat_SR[k*250:(k+1)*250,:,:,:] = sr1
                    Test_pred_mat_Limg[k*250:(k+1)*250:,:,:] = hr2
                    save_list = [sr1]
                    if (self.args.nmodels == 2):
                        sr2 = utility.quantize(sr2, self.args.rgb_range)
                        # print ("After quantize", sr.dtype)
                        # print ("After quantize", sr.max(),sr.min())
                        Test_pred_mat_HR_2[k*250:(k+1)*250,:,:,:] = hr2
                        Test_pred_mat_LR_2[k*250:(k+1)*250,:,:,:] = lr2
                        Test_pred_mat_SR_2[k*250:(k+1)*250,:,:,:] = sr2
                        #Test_pred_mat_Limg_2[k,:,:,:] = hr2

                        #print ("TEST-Data shape-LR",lr.size())
                        #print ("TEST-Data shape-HR",hr.size())
                        #print ("TEST-Data shape-SR",sr.size())
                    
                        save_list2 = [sr2]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr1, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])
                        if (self.args.nmodels == 2):
                            eval_acc2 += utility.calc_psnr(
                                sr2, hr2, scale, self.args.rgb_range,
                                benchmark=self.loader_test.dataset.benchmark
                            )
                            #print ("eval_acc",eval_acc,"eval_acc2",eval_acc2)                        
                            save_list2.extend([lr2, hr2])

                    if self.args.save_results:                 
                        self.ckp.save_results(filename, save_list, scale)
                        if (self.args.nmodels == 2):
                            self.ckp2.save_results(filename, save_list2, scale)
                    k=k+1
                dir_save = '/gpfs/jlse-fs0/users/sand33p/stronglensing/Image_Enhancement/EDSR_MWCNN/experiment/'+self.args.save
                filename_save_HR = '{}/results/Array_HR.npz'.format(dir_save) 
                filename_save_LR = '{}/results/Array_LR.npz'.format(dir_save) 
                filename_save_SR = '{}/results/Array_SR.npz'.format(dir_save) 
                filename_save_Limg = '{}/results/Array_Limg.npz'.format(dir_save) 

                np.savez_compressed(filename_save_HR, X=Test_pred_mat_HR)
                np.savez_compressed(filename_save_LR, X=Test_pred_mat_LR)
                np.savez_compressed(filename_save_SR, X=Test_pred_mat_SR) 
                np.savez_compressed(filename_save_Limg, X=Test_pred_mat_Limg) 
                if not self.args.test_only:
                    print ("self.it_ckp",self.it_ckp)
                    with h5py.File(self. HDF5_file_model1_loc, 'a')  as hf:   
                        hf["Array_HR"][self.it_ckp,:,:,:,:] = Test_pred_mat_HR
                        hf["Array_LR"][self.it_ckp,:,:,:,:] = Test_pred_mat_LR
                        hf["Array_SR"][self.it_ckp,:,:,:,:] = Test_pred_mat_SR
                        hf["Array_Limg"][self.it_ckp,:,:,:,:] = Test_pred_mat_Limg
                        hf.close()


                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR-ckp: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

                if (self.args.nmodels == 2):
                    dir_save = '/gpfs/jlse-fs0/users/sand33p/stronglensing/Image_Enhancement/EDSR_MWCNN/experiment/'+self.args.save
                    filename_save_HR_2 = '{}/results/Array_HR_2.npz'.format(dir_save) 
                    filename_save_LR_2 = '{}/results/Array_LR_2.npz'.format(dir_save) 
                    filename_save_SR_2 = '{}/results/Array_SR_2.npz'.format(dir_save) 
                    #filename_save_Limg_2 = '{}/results/Array_Limg_2.npz'.format(dir_save) 

                    np.savez_compressed(filename_save_HR_2, X=Test_pred_mat_HR_2)
                    np.savez_compressed(filename_save_LR_2, X=Test_pred_mat_LR_2)
                    np.savez_compressed(filename_save_SR_2, X=Test_pred_mat_SR_2) 

                    if not self.args.test_only:
                        with h5py.File(self. HDF5_file_model2_loc, 'a') as hf2:   
                            hf2["Array_HR"][self.it_ckp,:,:,:,:] = Test_pred_mat_HR_2
                            hf2["Array_LR"][self.it_ckp,:,:,:,:] = Test_pred_mat_LR_2
                            hf2["Array_SR"][self.it_ckp,:,:,:,:] = Test_pred_mat_SR_2                    
                            hf2.close()
                        

                    self.ckp2.log[-1, idx_scale] = eval_acc2/ len(self.loader_test)
                    best2 = self.ckp2.log.max(0)
                    self.ckp2.write_log(
                        '[{} x{}]\tPSNR-ckp2: {:.3f} (Best: {:.3f} @epoch {})'.format(
                                                self.args.data_test,
                                                scale,
                                                self.ckp2.log[-1, idx_scale],
                                                best2[0][idx_scale],
                                                best2[1][idx_scale] + 1
                                            )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if (self.args.nmodels == 2):
            self.ckp2.write_log(
                'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
            )
        if not self.args.test_only:
            #save(self, trainer, epoch, is_best=False,model=trainer.model1,loss=trainer.loss1,optimizer=trainer.optimizer1,model_name='model1'):
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch),model=self.model1,loss=self.loss1,optimizer=self.optimizer1,model_name='model1')
            if ((self.args.nmodels == 2) & (self.args.numloss == 2)):
                if(self.use_two_opt):
                    self.ckp2.save(self, epoch, is_best=(best2[1][0] + 1 == epoch),model=self.model2,loss=self.loss2,optimizer=self.optimizer2,model_name='model2')
                else:
                    self.ckp2.save(self, epoch, is_best=(best2[1][0] + 1 == epoch),model=self.model2,loss=self.loss2,optimizer=self.optimizer1,model_name='model2')
            #elif (self.args.nmodels == 2):
            #        self.ckp2.save(self, epoch, is_best=(best2[1][0] + 1 == epoch),model=self.model2,loss=self.loss1,optimizer=self.optimizer1,model_name='model2')
        if not self.args.test_only:
            self.it_ckp = self.it_ckp+1

    def prepare(self, l, volatile=False):
        #def _prepare(idx, tensor):
        #    if not self.args.cpu: tensor = tensor.cuda()
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()

        #    # Only test lr can be volatile
        #    return Variable(tensor, volatile=(volatile and idx==0))

            return tensor.to(device)  
           
        #return [_prepare(i, _l) for i, _l in enumerate(l)]
        return [_prepare(_l) for _l in l]

    def prepare_test(self, l, volatile=False):
        #def _prepare(idx, tensor):
        #    if not self.args.cpu: tensor = tensor.cuda()
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()

            # crop the odd dimension
            pad = torch.nn.ConstantPad2d((1,0,1,0),0)
            tensor = pad(tensor)
        #    # Only test lr can be volatile
        #    return Variable(tensor, volatile=(volatile and idx==0))

            return tensor.to(device)  
           
        #return [_prepare(i, _l) for i, _l in enumerate(l)]
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler1.last_epoch + 1
            return epoch >= self.args.epochs
