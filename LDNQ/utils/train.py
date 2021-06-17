import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim

import os
import time
import sys
from datetime import datetime
import numpy as np

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 25.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / float(self.count)


def adjust_mean_variance(net, dataset_loader, n_batch_used = 100, dataset_name ='CIFAR10', use_cuda = True):

    net.train()

    monitor_freq = 10

    if dataset_name == 'CIFAR10' or dataset_name == 'MNIST':
        correct = 0
        total = 0
    elif dataset_name == 'ImageNet':
        # losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(dataset_loader):

        if use_cuda:
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        else:
            inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs)

        if dataset_name == 'CIFAR10' or dataset_name == 'MNIST':
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % monitor_freq == 0:
                print ('[%d] Accuracy: %.3f' %(batch_idx, 100.*correct/total ))

            if batch_idx == n_batch_used:
                return

        elif dataset_name == 'ImageNet':
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            # losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            if batch_idx % monitor_freq == 0:
                print('Adjust: [{0}/{1}]\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(dataset_loader), top1=top1, top5=top5))
            if batch_idx == n_batch_used:
                return



def train(net, trainloader, optimizer, criterion = nn.CrossEntropyLoss(),
          dataset_name = 'CIFAR10', val_loader = None, use_cuda = True):

    if dataset_name == 'ImageNet':
        val_freq = 500
        n_epoch = 1000
        last_top1 = 0
        last_top5 = 0
    else:
        val_freq = 10
        n_epoch = 10
        best_acc = 0

    net.train()
    if dataset_name != 'ImageNet':
        train_loss = 0
        correct = 0
        total = 0
    else:
        monitor_freq = 100
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

    total_batch_idx = 0
    for epoch in range(n_epoch):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            total_batch_idx += 1
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            # print(targets.data)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if dataset_name != 'ImageNet':
                train_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            else:
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.data.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                '''progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Top1 acc: %.3f%%(%d/%d), Top5 acc: %.3f%%(%d/%d)'
                             % (losses.avg,
                                100. * top1.avg, top1.sum, top1.count,
                                100. * top5.avg, top5.sum, top5.count))'''
                if batch_idx % monitor_freq == 0:
                    print('Training: [{0}/{1}]\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        batch_idx, len(trainloader), top1=top1, top5=top5))

            if val_loader is not None and dataset_name == 'ImageNet' and (total_batch_idx + 1) % val_freq == 0:
                top1_acc, top5_acc = validate(net=net, validate_loader=val_loader, dataset_name=dataset_name)
                if np.abs(last_top1 - top1_acc) < 1 and np.abs(last_top5 - top5_acc) < 1:
                    return
                last_top1 = top1_acc
                last_top5 = top5_acc


        if dataset_name == 'CIFAR10' and val_loader is not None:
            print('[Validation] [Epoch: %d]' % epoch)
            acc = validate(net=net, validate_loader=val_loader, dataset_name=dataset_name)
            if best_acc < acc:
                best_acc = acc
    if dataset_name == 'CIFAR10':
        print ('Best acc: %.3f' %best_acc)


def validate(net, validate_loader, criterion = nn.CrossEntropyLoss(), n_batch_used = None,
             dataset_name = 'CIFAR10', val_record = None, use_cuda = True):

    net.eval()
    monitor_freq = 100
    if  dataset_name != 'ImageNet':
        test_loss = 0
        correct = 0
        total = 0
    else:
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(validate_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if dataset_name != 'ImageNet':
            '''if torch.__version__[2] == '3':
                test_loss += loss.data[0]
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()
            else:'''
            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

            progress_bar(batch_idx, len(validate_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' \
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        else:
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            if (batch_idx + 1) % monitor_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx + 1, len(validate_loader), top1=top1, top5=top5))

        if n_batch_used is not None and batch_idx == n_batch_used:
            break

    if dataset_name != 'ImageNet':
        if val_record is not None:
            val_record.write('Validation accuracy: %.3f\n' %(100. * correct / total))
        return 100. * correct / total
    else:
        if val_record is not None:
            val_record.write('Validation accuracy: Top1: %.3f, Top5: %.3f\n' %(top1.avg, top5.avg))
        return top1.avg, top5.avg


def cascade_soft_update(net, original_net, loader, optimizer, \
    criterion = nn.MSELoss(), dataset_name = 'CIFAR10', error_gap = 0.0001, \
    n_batch_used = None, train_record = None, use_cuda = True):
    """
    This function performs the core functionality of cascaded weights update: Given the quantized / original network and
    trainabale paramters (incorporated in optimizer), it performs training on those trainabale parameters.
    :param net:
    :param original_net:
    :param loader:
    :param optimizer:
    :param criterion:
    :param dataset_name:
    :param error_gap:
    :param n_batch_used:
    :param train_record:
    :param use_cuda:
    :return:
    """

    n_max_epoch = 100000000
    net.train()
    original_net.train()
    if dataset_name == 'ImageNet':
        moniter_freq = 10 # Modified according to original implementation
        # error_gap = 10e-4
    else:
        moniter_freq = 10
        # error_gap = 10e-10
    train_loss = 0
    last_loss = 0
    ascent_count = 0
    losses = AverageMeter()

    if dataset_name == 'ImageNet':
        top1 = AverageMeter()
        top5 = AverageMeter()

    for epoch in range(n_max_epoch):
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                original_inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
                with torch.no_grad():
                    quantized_inputs = Variable(inputs.cuda())
            else:
                original_inputs, targets = Variable(inputs), Variable(targets)
                with torch.no_grad():
                    quantized_inputs = Variable(inputs)

            optimizer.zero_grad()
            original_output = original_net(original_inputs)
            quantized_output = net(quantized_inputs)

            original_output = Variable(original_output.data)
            loss = criterion(quantized_output, original_output)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            losses.update(loss.data.item(), inputs.size(0))

            if dataset_name == 'ImageNet':
                prec1, prec5 = accuracy(quantized_output.data, targets.data, topk=(1, 5))
                # losses.update(loss.data[0], inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

            if batch_idx % moniter_freq == 0:
                # avg_train_loss = train_loss / (epoch * len(loader) + batch_idx + 1)
                if dataset_name == 'ImageNet':
                    print ('[%s] [%3d] [batch idx: %d / %d] Cascade training avg loss: %f, Top1 acc: %.3f, Top5 acc: %.3f' \
                          % (datetime.now(), epoch, batch_idx, len(loader), losses.avg, top1.avg, top5.avg))
                    if train_record is not None:
                        train_record.write(
                            '[%d] [batch idx: %d] Cascade training avg loss: %f, Top1 acc: %.3f, Top5 acc: %.3f\n' \
                            % (epoch, batch_idx, losses.avg, top1.avg, top5.avg))
                else:
                    print ('[%s] [%3d] [batch idx: %d / %d] Cascade training avg loss: %f' \
                          % (datetime.now(), epoch, batch_idx, len(loader), losses.avg))
                    if train_record is not None:
                        train_record.write('[batch idx: %d] [%3d] Cascade training avg loss: %f\n' \
                                           % (batch_idx, epoch, losses.avg))


                if n_batch_used is not None and batch_idx == n_batch_used:
                    # train_record.close()
                    return

                if losses.avg > last_loss or np.abs(losses.avg - last_loss) < error_gap:
                    ascent_count += 1
                else:
                    ascent_count = 0

                last_loss = losses.avg

                if ascent_count >= 3:
                    print ('Early stop for ascending 3 times.')
                    if train_record is not None:
                        train_record.write('Early stop for ascending 3 times.\n')
                    return