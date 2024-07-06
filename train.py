import os
import sys
import time
import torch
import torch.nn
import numpy as np
import argparse
from PIL import Image
from tensorboardX import SummaryWriter

# from validate import validate
from dataset.dataset import SyntheticImagesDataset
# from data import create_dataloader
from training.earlystop import EarlyStopping
from networks.trainer import Trainer
from options.options import TrainOptions

from utils.util import set_random_seed

from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.isVal = True
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    set_random_seed()
    opt = TrainOptions().parse()
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    val_opt = get_val_opt()
    from dataset.process import processing
    train_set = SyntheticImagesDataset(data_paths="/home/zhainaixin/hades/data/gan/gan_train", opt=opt, process_fn=processing)
    val_set = SyntheticImagesDataset(data_paths="/home/zhainaixin/hades/data/gan/gan_val", opt=val_opt, process_fn=processing)
    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=opt.batch_size,
                                                    shuffle=True,
                                                    num_workers=opt.numThreads)
    val_dataloader = torch.utils.data.DataLoader(val_set,
                                                    batch_size=opt.batch_size,
                                                    shuffle=False,
                                                    num_workers=opt.numThreads)


    # data_loader = create_dataloader_new(opt)
    dataset_size = len(train_dataloader)
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystopEpoch, delta=-0.001, verbose=True)
    
    # opt = get_processing_model(opt)
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataloader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')
            # print("Iter time: %d sec" % (time.time()-iter_data_time))
            # iter_data_time = time.time()
            
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()

        y_true, y_pred = [], []
        val_i = 0 
        for img, label, paths in val_dataloader:
            val_i += 1
            print(f"batch number {val_i}/{len(val_dataloader)}", end="\r")
            in_tens = img.cuda()
            y_pred.extend(model.model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
            
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        acc = accuracy_score(y_true, y_pred>0.5)
        ap = average_precision_score(y_true, y_pred)
        
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystopEpoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()