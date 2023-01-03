import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import LDAMLoss, FocalLoss
import wandb
from hessian import hessian
import math
import numpy as np
import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet32)')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                     dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--log_results',action ='store_true')
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--loss_type_hess', type=str, default='CE')
parser.add_argument('--dataloader_hess', type=str, default='train')
parser.add_argument('--train_rule_hess', type=str, default='None')
parser.add_argument('--numpy_load', type=str, default='None')
best_acc1 = 0


def main():
    args = parser.parse_args()
    name = args.resume.split('/')[-2]
    args.store_name = '_'.join(['hessian',name,args.dataloader_hess, args.loss_type_hess, args.train_rule_hess, args.exp_str])
    print("_________")
    print("The store name is ", args.store_name)
    print("_________")
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    if args.log_results:
        wandb.init(project="long-tail",
                                   entity="active-learning", name=args.store_name)
        wandb.config.update(args)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes =  10
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            print("best acc:" , best_acc1)
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    model.eval()

    # Data loading code

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    
    train_sampler = None
        
    if args.train_rule_hess=="None":
        per_cls_weights=None
    elif args.train_rule_hess=="DRW":
        print("Using DRW")
        idx = 1
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
    else:
        print("Train Rule not supported")
    if args.loss_type_hess == 'CE':
        print("Using CE loss")
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
    elif args.loss_type_hess == 'LDAM':
        print("Using LDAM Loss")
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)

    if args.dataloader_hess=="train":
        print("Calculating Hessian on Train dataset")
        hess_dataset = train_dataset
    elif args.dataloader_hess=="val":
        print("Calculating Hessian on Val dataset")
        hess_dataset = val_dataset

    all_trace = []
    all_eigenvalues = []
    print('clas_num_list',cls_num_list)
    for class_idx, num_samples in enumerate(cls_num_list):
        if args.dataloader_hess=="train":
            print(f"Number of samples in class{class_idx}: {num_samples}")
        cls_indices = [i for i in range(len(hess_dataset.targets)) if hess_dataset.targets[i]==class_idx]
        class_idx_dataset = torch.utils.data.Subset(hess_dataset, cls_indices)
        if args.dataloader_hess=="val":
            print(f"Number of samples in class{class_idx}: {len(class_idx_dataset)}")
        class_loader = torch.utils.data.DataLoader(
                class_idx_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        validate(class_loader, model, criterion, None, args, log=None, tf_writer=None, class_idx=class_idx)

        hessian_comp = hessian(model, criterion, dataloader=class_loader, cuda=True)
        top_eigenvalues, _ = hessian_comp.eigenvalues()
        trace = hessian_comp.trace()
        density_eigen, density_weight = hessian_comp.density()
        save_numpy(args, class_idx, density_eigen, density_weight)



        print("The type of density_eigen is ", type(density_eigen))
        print("The type of density_weight is ", type(density_weight))
        print("The density eigen ", density_eigen)
        print("The density weight ", density_weight)
        print('\n***Top Eigenvalues: ', top_eigenvalues)
        print('\n***Trace: ', np.mean(trace))
        all_trace.append(np.mean(trace))
        all_eigenvalues.append(top_eigenvalues[0])
        get_esd_plot(density_eigen, density_weight, class_idx)
    
    print("Trace:", all_trace)
    print("Eigenvalues:", all_eigenvalues)


    return

def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val', class_idx=-1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        if args.log_results and class_idx==-1:
            if args.dataset=="cifar10":
                for idx in range(len(cls_acc)):
                    wandb.log({f"class{idx}_acc":cls_acc[idx]})
            if args.dataset=="cifar100":
                wandb.log({f"class0_acc":cls_acc[0]})
                for idx in range(1,5):
                    wandb.log({f"class{100-idx}_acc":cls_acc[-idx]})
                    wandb.log({f"class{idx}_acc":cls_acc[idx]})
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        if args.log_results and class_idx!=-1:
            wandb.log({f'class{class_idx}_loss':losses.avg})
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()
        if tf_writer is not None:
            tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
            tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_esd_plot(eigenvalues, weights, class_idx):
    density, grids = density_generate(eigenvalues, weights)
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
    plt.xlabel('Eigenvalue', fontsize=14, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.tight_layout()
    lambda_min = np.min(eigenvalues)
    lambda_max = np.max(eigenvalues)
    lambda_ratio = lambda_max/lambda_min


    wandb.log({"esd plot{class_idx}":plt})
    wandb.log({f"lambda_min-{class_idx}":lambda_min, f"lambda_max-{class_idx}":lambda_max, f"lambda_ratio-{class_idx}":lambda_ratio})
    #plt.savefig('example.pdf')


def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.01):

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)

if __name__ == '__main__':
    main()
