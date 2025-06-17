import time
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from scipy.stats import gmean

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboard_logger import Logger

from loss import *
from utils import *

import os
import numpy as np
import random
import torch.nn.init as init
from parser import get_parser
from dataset.datasets_aug_all_num_homo import AgeDB


os.environ["KMP_WARNINGS"] = "FALSE"


random_seed = 0 # or any of your favorite number 
torch.manual_seed(random_seed)
#torch.cuda.manual_seed_all(random_seed) 
#torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True)
#torch.backends.cudnn.benchmark = False
#print(f"Seed: {torch.seed()}")
#print(f"Manual Seed: {torch.manual_seed()}")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

#def init():
parser = get_parser()
args, unknown = parser.parse_known_args()
args.start_epoch, args.best_loss, args.best_acc = 0, 1e5, 0

fld = "_".join(args.exp.split("_")[1:])
if len(args.store_name):
    args.store_name = f'_{args.store_name}'
if not args.lds and args.reweight != 'none':
    args.store_name += f'_{args.reweight}'
if args.lds:
    args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
    if args.lds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.lds_sigma}'
if args.fds:
    args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
    if args.fds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.fds_sigma}'
    args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
if args.retrain_fc:
    args.store_name += f'_retrain_fc'
args.store_name = f"{args.dataset}_{args.model}{args.store_name}_{args.optimizer}_{args.loss}_{args.lr}_{args.batch_size}"

args.store_name = args.data_dir + "_" + args.store_name
args.exp = 'Exp_' + args.exp
if args.tem == 0:
    from network.neural_net import NeuralNetwork
if args.tem == 1:
    from network.neural_net_temp import NeuralNetwork
prepare_folders(args)

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.exp, args.store_name, 'training_.log')),
        logging.StreamHandler()
    ])
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")

tb_logger = Logger(logdir=os.path.join(args.store_root, args.exp, args.store_name), flush_secs=2)
#return args, tb_logger, fld

#args, tb_logger, fld = init()


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter(f'Loss ({args.loss.upper()})', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    for idx, (inputs, targets, weights) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs, targets, weights = \
            inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True), weights.cuda(non_blocking=True)
        if args.fds:
            outputs, _ = model(inputs)#, targets, epoch)
        else:
            outputs = model(inputs)#, targets, epoch)
        loss = globals()[f"weighted_{args.loss}_loss"](outputs, targets, weights)
        assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            progress.display(idx)

    if args.fds and epoch >= args.start_update:
        print(f"Create Epoch [{epoch}] features of all training data...")
        encodings, labels = [], []
        with torch.no_grad():
            for (inputs, targets, _) in tqdm(train_loader):
                inputs = inputs.cuda(non_blocking=True)
                outputs, feature = model(inputs)#, targets, epoch)
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(targets.data.squeeze().cpu().numpy())

        encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(np.hstack(labels)).cuda()
        #model.module.FDS.update_last_epoch_stats(epoch)
        #model.module.FDS.update_running_stats(encodings, labels, epoch)

    return losses.avg

def get_predictions(loss_c, preds, labels, files, val=False):
    train_s = ""
    if val:
        train_s = " train"
    if loss_c == "mse":
        paths = files
        output = (np.abs(np.array([round(i.item()) for i in preds*10])-np.array([round(i.item()) for i in labels*10]))).tolist()
        files = [(i,j,k) for i,j,k in zip([round(i.item()) for i in labels*10],[p for p in paths],[round(i.item()) for i in preds*10])]
        acc_1 = len([i for i in output if i<=1])/len(output)
        print("Accuracy" + train_s + " +-0 " + str(len([i for i in output if i<=0])/len(output)))
        print("Accuracy" + train_s + " +-1 " + str(acc_1))
        print("Accuracy" + train_s + " +-2 " + str(len([i for i in output if i<=2])/len(output)))
        print("Accuracy" + train_s + " +-3 " + str(len([i for i in output if i<=3])/len(output)))
        if not val:
            for i,j,k in files:
                print(str(k) + " | " + str(i) + " | " + str(j))
    

    return acc_1

def validate(val_loader, model, train_labels=None, prefix='Val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    losses_all = []
    preds, labels, files = [], [], []
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _, fls) in enumerate(val_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            #outputs = torch.argmax(outputs, dim=1)
            #outputs = (outputs +1)/10
            #import ipdb
            #ipdb.set_trace()
            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())
            files.extend(fls)
            
            loss_mse = criterion_mse(outputs, targets)
            loss_l1 = criterion_l1(outputs, targets)
            loss_all = criterion_gmean(outputs, targets)
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)
        loss_gmean = 0
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        acc_1 = get_predictions("mse", preds, labels, files)
    return losses_mse.avg, losses_l1.avg, loss_gmean, acc_1
    #return 0, 0, 0, acc_1


def validate_train(val_loader, model, train_labels=None, prefix='Val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    losses_all = []
    preds, labels, files = [], [], []
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _, fls) in enumerate(val_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            
            #outputs = torch.argmax(outputs, dim=1)
            #outputs = (outputs +1)/10

            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())
            files.extend(fls)
            
            loss_mse = criterion_mse(outputs, targets)
            loss_l1 = criterion_l1(outputs, targets)
            loss_all = criterion_gmean(outputs, targets)
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)
        loss_gmean = 0
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        acc_1 = get_predictions("mse", preds, labels, files, val=True)
        #for i,j,k in files:
        #    print(str(k) + " | " + str(i) + " | " + str(j))
    return losses_mse.avg, losses_l1.avg, loss_gmean, acc_1
    #return 0,0,0,acc_1

def main():
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Data
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    train_dataset = AgeDB(data_dir=args.data_dir, df=None, img_size=args.img_size, group=fld, split='train',
                          reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
    train_labels = train_dataset.labels
    val_dataset = AgeDB(data_dir=args.data_dir, df=None, img_size=args.img_size, group=fld, split='val')
    test_dataset = AgeDB(data_dir=args.data_dir, df=None, img_size=args.img_size, group=fld, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Model
    print('=====> Building model...')
    
    
    
    model = NeuralNetwork(args)
    model = model.to("cuda")
    for ind, (name, para) in enumerate(model.named_parameters()):
        if ind < 0:
            para.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(train_loader, model, optimizer, epoch)
        val_loss_mse, val_loss_l1, val_loss_gmean, acc_1 = validate(val_loader, model, train_labels=train_labels)
        validate_train(test_loader, model, train_labels=train_labels)

        if epoch > 0:
            is_best = args.best_acc < acc_1
            args.best_acc = max(acc_1, args.best_acc)
        else:
            is_best = False
            args.best_acc = 0
        print(f"Best Accuracy: {args.best_acc} and Cur Acc: {acc_1}" )
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_acc': args.best_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
        print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
              f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]")

        tb_logger.log_value('train_loss', train_loss, epoch)
        tb_logger.log_value('val_loss_mse', val_loss_mse, epoch)
        tb_logger.log_value('val_loss_l1', val_loss_l1, epoch)
        tb_logger.log_value('val_loss_gmean', val_loss_gmean, epoch)
if __name__ == '__main__':
    main()
