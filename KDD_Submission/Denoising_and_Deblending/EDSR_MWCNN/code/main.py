import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import time

start_time = time.time()
torch.manual_seed(args.seed)

checkpoint = utility.checkpoint(args,'model1')
if (args.nmodels == 2):
    checkpoint2 = utility.checkpoint(args,'model2')
    
if checkpoint.ok:
    loader = data.Data(args)
    model1 = model.Model(args, checkpoint,'model1')
    loss1 = loss.Loss(args, checkpoint,'model1') if not args.test_only else None    
    if (args.nmodels == 2):
        model2 = model.Model(args, checkpoint2,'model2')
        loss2 = loss.Loss(args, checkpoint2,'model2') if not args.test_only else None
        print ("calling 2 model trainer")
        t = Trainer(args, loader, model1, loss1, checkpoint, model2,  loss2, checkpoint2)
    else:
        t = Trainer(args, loader, model1, loss1, checkpoint)
    while not t.terminate():
        current_epoch = t.scheduler1.last_epoch+1
        t.train()
        if ((current_epoch % 10)==0):
            t.test()

    checkpoint.done()

print("---Run time  %s seconds ---" % (time.time() - start_time))

