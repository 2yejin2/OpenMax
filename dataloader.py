import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def train_loader(data_root, data, batch_size):
    data_path = os.path.join(data_root, 'train', data)

    mean = [0.507, 0.487, 0.441]
    stdv = [0.267, 0.256, 0.276]

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
    ])

    train_dataset = datasets.ImageFolder(data_path,
                                         transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    return train_loader


def test_loader(data_root, data, batch_size, mode, transform=False):
    if mode == 'valid':
        data_path = os.path.join(data_root, 'valid', data, 'labels')
    elif mode == 'test':
        data_path = os.path.join(data_root, 'test', data, 'labels')

    mean = [0.507, 0.487, 0.441]
    stdv = [0.267, 0.256, 0.276]

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=stdv)])
    if transform == True:
        test_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=stdv)])

    test_dataset = datasets.ImageFolder(data_path,
                                        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    test_targets = test_dataset.targets

    return test_loader, test_targets


def in_dist_loader(data_root, data, batch_size, mode, transform=False):
    if mode == 'valid':
        data_path = os.path.join(data_root, 'valid', data, 'no-labels')
    elif mode == 'test':
        data_path = os.path.join(data_root, 'test', data, 'no-labels')

    mean = [0.507, 0.487, 0.441]
    stdv = [0.267, 0.256, 0.276]

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean,
                                                              std=stdv)])
    if transform == True:
        test_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=stdv)])

    in_dataset = datasets.ImageFolder(data_path,
                                      transform=test_transform)

    in_loader = torch.utils.data.DataLoader(in_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4)

    return in_loader


def out_dist_loader(data_root, data, batch_size, mode, transform=False):
    data_path = os.path.join(data_root, mode, data, 'no-labels')

    mean = [0.507, 0.487, 0.441]
    stdv = [0.267, 0.256, 0.276]

    if data == 'new-tinyimagenet158':
        test_transform = transforms.Compose([transforms.Resize((32, 32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean,
                                                                  std=stdv)])

        if transform == True:
            test_transform = transforms.Compose([transforms.Resize((32, 32)),
                                                 transforms.RandomCrop((32, 32), padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=mean,
                                                                      std=stdv)])
    else:
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean,
                                                                  std=stdv)])
        if transform == True:
            test_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=mean,
                                                                      std=stdv)])

    out_dataset = datasets.ImageFolder(data_path,
                                       transform=test_transform)

    out_loader = torch.utils.data.DataLoader(out_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4)

    return out_loader
