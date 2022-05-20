import resnet

import dataloader
import utils
import train
import plot

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description='Misclassification Detection / Out of Distribution Detection / Open Set Recognition')

parser.add_argument('--epochs', default=200, type=int, help='epoch (default: 200)')
parser.add_argument('--batch-size', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--model', default='res18', type=str, help='models : res18 (default: res18)')
parser.add_argument('--data', default='cifar40', type=str, help='datasets : cifar100 / tinyimagenet / svhn / LSUN (default: cifar100)')
parser.add_argument('--data-root', default='/daintlab/data/md-ood-osr/data', type=str, help='data path')
parser.add_argument('--save-root', default='./real_final/', type=str, help='save root')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--trial', default='03', type=str)
parser.add_argument('--gpu-id', default='0', type=str, help='gpu number')

args = parser.parse_args()

def main():
    # Set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cudnn.benchmark = True

    # Make folder and save path
    save_path = os.path.join(args.save_root, f'trial-{args.trial}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Make Data set
    train_loader = dataloader.train_loader(args.data_root,
                                           args.data,
                                           args.batch_size)
    test_loader, test_targets = dataloader.test_loader(args.data_root,
                                                       args.data,
                                                       args.batch_size,
                                                       mode = 'test')

    # Set network
    if args.model == 'res18':
        net = resnet.ResNet18(num_classes=40).cuda()

    # Set criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # Set optimizer (default:sgd)
    optimizer = optim.SGD(net.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=0.0005,
                          nesterov=True)

    # Set scheduler
    scheduler = MultiStepLR(optimizer,
                            milestones=[120,160],
                            gamma=0.1)

    # Make logger
    train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
    test_logger = utils.Logger(os.path.join(save_path, 'test.log'))

    # Start Train
    for epoch in range(1, args.epochs+1):
        # Train
        train.train(train_loader,
                    net,
                    criterion,
                    optimizer, epoch,
                    train_logger,
                    args)

        # Validation
        train.valid(test_loader,
                    net,
                    criterion,
                    epoch,
                    test_logger,
                    'test',
                    args)
        # scheduler
        scheduler.step()
    # Finish Train

    # Save Model
    torch.save(net.state_dict(),
               os.path.join(save_path, f'model_{int(args.epochs)}.pth'))

    # Draw Plot
    plot.draw_plot(save_path)


if __name__ == "__main__":
    main()
