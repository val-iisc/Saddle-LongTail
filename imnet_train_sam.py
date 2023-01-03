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
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from models import resnet
from models.resnet_cifar import NormedLinear
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from datasets.imagenet import ImageNet_LT
from losses import LDAMLoss, FocalLoss
import wandb
from sam import SAM
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50'),
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='additional info to indicate which experiment it is')
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--log_results', action='store_true',
                    help='to log results on wandb')
parser.add_argument('--rho', type=float, default=0.05)
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--distributed', action='store_true',
                    help='use distributed model')
parser.add_argument('--deterministic', action='store_true',
                    help='use deterministic')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='use deterministic')
parser.add_argument('--data_path', default='Imagenet', type=str, metavar='PATH',
                    help='path to latest dataset ')
parser.add_argument('--cos_lr', action='store_true',
                    help='Using cosine lr')
parser.add_argument('--end_lr_cos', default=0.0, type=float, metavar='M',
                    help='End lr for cos learning schedule')
parser.add_argument('--min_rho', type=float, default=0.05)
parser.add_argument('--max_rho', type=float, default=0.8)
parser.add_argument('--rho_schedule', default='none',
                    choices=('none','linear', 'step'))
parser.add_argument('--rho_steps', metavar = 'N', type = float, nargs = '+', help = 'rho values at various steps')
parser.add_argument('--margin', default=0.5, type=float, metavar='M',
                    help='Margin value for LDAM')

best_acc1 = 0


def main():
    args = parser.parse_args()
    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor),'sam','rho_step', '-'.join(map(str, args.rho_steps)), args.exp_str])
    print("The args.store name is", args.store_name)
      
      
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
    args.head_class_idx = [0,390]
    args.med_class_idx = [390,835]
    args.tail_class_idx = [835,1000]
    if args.log_results:
        wandb.init(project="long-tail",
                                   entity="long-tail", name=args.store_name)
        wandb.config.update(args)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("[INFORMATION] creating model '{}'".format(args.arch))
    num_classes = 1000 
    use_norm = True if args.loss_type == 'LDAM' else False
    if args.arch == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False) 
    else:
        warnings.simplefilter("error") 
        warnings.warn("Add support for other models apart from resnet50")
    if use_norm:
      model.fc = NormedLinear(2048, num_classes)
      print("[INFORMATION] Using normed linear")
    else:
      model.fc = nn.Linear(2048, num_classes)
    model = model.cuda(args.gpu)
        # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

    base_optimizer = torch.optim.SGD
    optimizer = SAM(base_optimizer=base_optimizer, rho=args.rho, params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.cos_lr == True:
      print("[INFORMATION] Using cosine lr_scheduler")
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.end_lr_cos)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("[INFORMATION] loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.cos_lr == True:
              scheduler.load_state_dict(checkpoint['scheduler'])
            print("[INFORMATION] loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("[INFORMATION] no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

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

    if args.dataset == 'imagenet':
      dataset = ImageNet_LT(args.distributed, root=args.data_path,
                              batch_size=args.batch_size, num_works=args.workers)
      args.cls_num_list = dataset.cls_num_list
      cls_num_list = dataset.cls_num_list
      print("The class list for imagenet(initial 20) is ", dataset.cls_num_list[:20])
      print("The class list for imagenet(last 20) is ", dataset.cls_num_list[-20:])
    else:
        warnings.warn('Dataset is not listed')
        return
    
    train_sampler = None
        
    train_loader = dataset.train_instance 
    val_loader = dataset.eval

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_rho(optimizer, epoch, args)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            rho = param_group['rho']
        if args.log_results:
            wandb.log({'lr':lr, 'rho':rho, 'epoch':epoch})

        if args.train_rule == 'None':
            train_sampler = None  
            per_cls_weights = None 
        elif args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 60
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')
        
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            print("[INFORMATION] LDAM is being used")
            print("[INFORMATION] margin value being used is ", args.margin)
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=args.margin, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)
        if args.cos_lr == True:
          scheduler.step()
        
        if args.log_results:
            wandb.log({'epoch':epoch, 'val_acc':acc1})
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()
        if args.cos_lr == True:
          save_checkpoint(args, {
              'epoch': epoch + 1,
              'arch': args.arch,
              'state_dict': model.state_dict(),
              'best_acc1': best_acc1,
              'optimizer' : optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
          }, is_best)
            
        else:

          save_checkpoint(args, {
              'epoch': epoch + 1,
              'arch': args.arch,
              'state_dict': model.state_dict(),
              'best_acc1': best_acc1,
              'optimizer' : optimizer.state_dict(),
          }, is_best)
    if args.log_results:
        wandb.log({'best_acc':best_acc1})

def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad=True)
        if args.log_results:
            wandb.log({'loss1':loss})

        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.second_step(zero_grad=True)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.log_results:
            wandb.log({'loss':loss, 'top1_acc':acc1, 'top5_acc':acc5})
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
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
            output = model(input) #bs, num_classes
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1) #bs
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
        print("[INFORMATION] The size of the cf is", cf.shape)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf) #num of correct preds
        cls_acc = cls_hit / cls_cnt

        if args.dataset == 'imagenet':
          
            head_acc = cls_acc[args.head_class_idx[0]:args.head_class_idx[1]].mean() * 100

            med_acc = cls_acc[args.med_class_idx[0]:args.med_class_idx[1]].mean() * 100
            tail_acc = cls_acc[args.tail_class_idx[0]:args.tail_class_idx[1]].mean() * 100
            print(f"The head accuracy is {head_acc}\n")
            print(f"The med accuracy is {med_acc}\n")
            print(f"The tail accuracy is {tail_acc}\n")
            if args.log_results:
              wandb.log({'head_acc':head_acc, 'med_acc':med_acc, 'tail_acc':tail_acc})
        
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        if args.dataset != 'imagenet':
          print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg


def adjust_rho(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if args.rho_schedule == 'step':
      if epoch <= 5:
          rho = args.rho_steps[0]
      elif epoch > 75:
          rho = args.rho_steps[3]
      elif epoch > 60:
          rho = args.rho_steps[2]

      else:
          rho = args.rho_steps[1]
      for param_group in optimizer.param_groups:
          param_group['rho'] = rho
    if args.rho_schedule == 'linear':
      X = [1, args.epochs]
      Y = [args.min_rho, args.max_rho]
      y_interp = interp1d(X, Y)
      rho = y_interp(epoch)

      for param_group in optimizer.param_groups:

          param_group['rho'] = np.float16(rho)
    if args.rho_schedule == 'none':
      rho = args.rho
      for param_group in optimizer.param_groups:
          param_group['rho'] = rho

if __name__ == '__main__':
    main()
