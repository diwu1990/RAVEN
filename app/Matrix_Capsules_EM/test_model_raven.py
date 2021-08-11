from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import capsules

from model.capsules_raven import CapsNet as CapsNetRAVEN
from loss import SpreadLoss
from datasets import smallNORB

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--test-intvl', type=int, default=1, metavar='N',
                    help='test intvl (default: 1)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=3e-3, metavar='LR', 
                    help='learning rate (default: 0.01)') # moein - according to openreview
parser.add_argument('--weight-decay', type=float, default=2e-7, metavar='WD',
                    help='weight decay (default: 0)') # moein - according to openreview
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--em-iters', type=int, default=2, metavar='N',
                    help='iterations of EM Routing')
parser.add_argument('--snapshot-folder', type=str, default='/home/diwu/Project/RAVEN/app/Matrix_Capsules_EM/snapshots', metavar='SF',
                    help='where to store the snapshots')
parser.add_argument('--data-folder', type=str, default='/mnt/ssd1/data', metavar='DF',
                    help='where to store the datasets')
parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                    help='dataset for training(mnist, smallNORB)')
# extra parameters for raven design
parser.add_argument('--cycle', type=int, default=8, metavar='C',
                    help='cycle count for nonlinear operation')
parser.add_argument('--intwidth-max', type=int, default=7, metavar='I',
                    help='maximum integer width')
parser.add_argument('--fracwidth-max', type=int, default=8, metavar='F',
                    help='maximum fracwidth width')
parser.add_argument('--bitwidth-reduce', action='store_true', default=False,
                    help='allows to reduce MAC bitwidth')
parser.add_argument('--rounding', type=str, default='round', metavar='R',
                    help='rounding mode')                                                       


def get_setting(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    path = os.path.join(args.data_folder)
    if args.dataset == 'mnist':
        num_class = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'smallNORB':
        num_class = 5
        train_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.RandomCrop(32),
                          transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          transforms.ToTensor()
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=False,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor()
                      ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise NameError('Undefined dataset {}'.format(args.dataset))
    return num_class, train_loader, test_loader


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
        self.avg = self.sum / self.count
		
		
def test(test_loader, model, criterion, device, min_cnt=False):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    
    if min_cnt is True:
        total_cnt = min(1, test_len)
    else:
        total_cnt = max(1, test_len)
    idx = 0
    with torch.no_grad():
        for data, target in test_loader:
            if idx < total_cnt:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target, r=1).item()
                acc += accuracy(output, target)[0].item()
                idx += 1
    
    test_loss /= test_len
    acc /= test_len
    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, acc))
    return acc


def main():
    global args, best_prec1
    args = parser.parse_args()

    print("Current setting: cycle-"     + str(args.cycle) + \
          "\tintwidth-max-"               + str(args.intwidth_max) + \
          "\tfracwidth-max-"              + str(args.fracwidth_max) + \
          "\tbitwidth-reduce-"            + str(args.bitwidth_reduce) + \
          "\trounding-"                   + args.rounding + "\n")

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    # datasets
    num_class, train_loader, test_loader = get_setting(args)

    # model
    A, B, C, D = 64, 8, 16, 16
    # A, B, C, D = 32, 32, 32, 32

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)

    print("Floating-point model:")
    model = capsules(A=A, B=B, C=C, D=D, E=num_class,
                    iters=args.em_iters).to(device)
    model.load_state_dict(torch.load(args.snapshot_folder+"/model_10.pth"))
    model_test_acc = test(test_loader, model, criterion, device)


    print("RAVEN model:")
    modelRAVEN = CapsNetRAVEN(A=A, B=B, C=C, D=D, E=num_class, iters=args.em_iters, 
                        cycle=args.cycle, 
                        intwidth=args.intwidth_max, 
                        fracwidth=args.fracwidth_max, 
                        bitwidth_reduce=args.bitwidth_reduce, 
                        rounding="round").to(device)
    modelRAVEN.eval()
    modelRAVEN_state_dict = modelRAVEN.state_dict()
    modelRAVEN_state_dict.update(torch.load(args.snapshot_folder+"/model_10.pth"))
    modelRAVEN.load_state_dict(modelRAVEN_state_dict)
    modelRAVEN_test_acc = test(test_loader, modelRAVEN, criterion, device)


if __name__ == '__main__':
    main()
