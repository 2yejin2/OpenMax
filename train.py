import utils
import metrics

import time

import torch
import torch.nn.functional as F

def train(loader, model, criterion, optimizer, epoch, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    end = time.time()

    model.train()
    for i, (input, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # set input ,target
        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)

        # for logit regularization
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec, correct = utils.accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

    logger.write([epoch, losses.avg, top1.avg])


def valid(loader, model, criterion, epoch, logger, mode, args):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    end = time.time()

    li_conf = []
    li_corr = []
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec, correct = utils.accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            probs = F.softmax(output, dim=1)
            confidence, _ = probs.max(dim=1)
            li_conf.extend(confidence)
            li_corr.extend(correct)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print(mode, ': [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                    i, len(loader), batch_time=batch_time, loss=losses,
                    top1=top1))

        conf_corr = sorted(zip(li_conf, li_corr), key=lambda x: x[0], reverse=True)
        sorted_conf, sorted_corr = zip(*conf_corr)
        li_risk, li_cov = metrics.coverage_risk(sorted_conf, sorted_corr)
        aurc, eaurc = metrics.calc_eaurc(li_risk)

        print('* Prec {top1.avg:.3f}% '.format(top1=top1))

        logger.write([epoch, losses.avg, top1.avg, aurc, eaurc])
