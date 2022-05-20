from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import argparse
import matplotlib
import numpy as np
matplotlib.use('agg')
import sklearn.covariance
from collections import Iterable

import metrics


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 'T', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'F', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class AverageMeter(object):
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

class Logger(object):
    def __init__(self, path, int_form=':04d', float_form=':.6f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.')
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)

        return log

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], correct.squeeze()


def get_values(loader, net, criterion):
    net.eval()

    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        accuracy = 0

        li_softmax = []
        li_max = []
        li_correct = []
        li_logit = []

        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()

            output = net(input)

            softmax = F.softmax(output, dim=1)
            conf, _ = torch.max(softmax.data, dim=1)
            li_max.extend(conf.cpu().data.numpy())

            loss = criterion(output, target).cuda()

            total_loss += loss.mean().item()
            pred = output.data.max(1, keepdim=True)[1]

            total_acc += pred.eq(target.data.view_as(pred)).sum()

            for i in output:
                li_logit.append(i.cpu().data.numpy())

            li_softmax.extend(F.softmax(output).cpu().data.numpy())

            for i in range(len(pred)):
                if pred[i] == target[i]:
                    accuracy += 1
                    cor = 1
                else:
                    cor = 0
                li_correct.append(cor)


        total_loss /= len(loader)
        total_acc = 100. * total_acc / len(loader.dataset)

        print('loss: {:.6f}     accuracy: {:.6f}%'.format(total_loss, total_acc))

    return li_softmax, li_correct, li_logit, np.array(li_max)


def get_values_openmax(loader, net, criterion):
    net.eval()

    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        accuracy = 0

        li_softmax = []
        li_max = []
        li_correct = []
        li_logit = []

        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()

            output = net(input)

            softmax = F.softmax(output, dim=1)
            conf, _ = torch.max(softmax.data, dim=1)
            li_max.extend(conf.cpu().data.numpy())

            loss = criterion(output, target).cuda()

            total_loss += loss.mean().item()
            pred = output.data.max(1, keepdim=True)[1]

            total_acc += pred.eq(target.data.view_as(pred)).sum()

            for i in output:
                li_logit.append(i.cpu().data.numpy())

            li_softmax.extend(F.softmax(output).cpu().data.numpy())

            for i in range(len(pred)):
                if pred[i] == target[i]:
                    accuracy += 1
                    cor = 1
                else:
                    cor = 0
                li_correct.append(cor)


        total_loss /= len(loader)
        total_acc = 100. * total_acc / len(loader.dataset)

        print('loss: {:.6f}     accuracy: {:.6f}%'.format(total_loss, total_acc))

    return li_softmax, li_correct, li_logit, np.array(li_max)



def get_posterior(model, test_loader, magnitude, temperature):

    criterion = nn.CrossEntropyLoss()
    stdv = [0.267, 0.256, 0.276]

    model.eval()
    total = 0
    accuracy = 0
    li_softmax = []
    li_correct = []
    li_mcp = []
    for j, (input, target) in enumerate(test_loader):
        total += input.size(0)
        input = input.cuda()
        input = Variable(input, requires_grad=True)

        batch_output = model(input)

        # temperature scaling
        outputs = batch_output / temperature
        labels = outputs.data.max(1)[1]
        labels = Variable(labels)
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(input.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / stdv[0])
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                             gradient.index_select(1, torch.LongTensor([1]).cuda()) / stdv[1])
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                             gradient.index_select(1, torch.LongTensor([2]).cuda()) / stdv[2])

        tempInputs = torch.add(input.data, -magnitude, gradient)
        outputs = model(Variable(tempInputs, volatile=True))
        outputs = outputs / temperature
        soft_out = F.softmax(outputs, dim=1)
        soft_out, pred = torch.max(soft_out.data, dim=1)
        li_mcp.append(soft_out.cpu().numpy())
        li_softmax.extend(F.softmax(outputs).cpu().data.numpy())

        for i in range(len(pred)):
            if pred[i] == target[i]:
                cor = 1
            else:
                cor = 0
            li_correct.append(cor)

    li_mcp = np.concatenate(li_mcp)

    return li_mcp, li_correct, li_softmax


def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def sample_estimator(model, num_classes, feature_list, train_loader):

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, volatile=True)
        output, out_features = model.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


def get_Mahalanobis_score(model, test_loader, num_classes, sample_mean, precision, layer_index, magnitude):

    model.eval()
    stdv = [0.267, 0.256, 0.276]

    Mahalanobis = []
    output = []
    labels = []
    for data, target in test_loader:

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target)

        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / stdv[0])
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                             gradient.index_select(1, torch.LongTensor([1]).cuda()) / stdv[1])
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                             gradient.index_select(1, torch.LongTensor([2]).cuda()) / stdv[2])
        tempInputs = torch.add(data.data, -magnitude, gradient)

        cls_outputs = model(Variable(tempInputs, volatile=True))
        soft_out = F.softmax(cls_outputs, dim=1)
        soft_out, pred = torch.max(soft_out.data, dim=1)
        labels.append(pred.cpu().numpy())

        noise_out_features = model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

        output.append(noise_gaussian_score.cpu().numpy())
        # labels.append(target.cpu().numpy())
    output = np.concatenate(output)
    labels = np.concatenate(labels)

    return output, labels


def detection_performance(regressor, X, Y):
    """
    Measure the detection performance
    return: detection metrics
    """
    num_samples = X.shape[0]
    y_pred = regressor.predict_proba(X)[:, 1]

    l1, l2 = [], []
    for i in range(num_samples):
        if Y[i] == 0:
            l1.append((-y_pred[i]))
        else:
            l2.append((-y_pred[i]))

    results = metrics.ood_metrics(np.array(l1), np.array(l2), mahal=True)

    return results, l1, l2
