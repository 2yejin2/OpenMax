import resnet
import openmax_utils
import dataloader
import metrics
import utils
import plot

import argparse
import os
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(
    description='Misclassification Detection / Out of Distribution Detection / Open Set Recognition')

parser.add_argument('--batch-size', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--model', default='res18', type=str, help='models: res18 (default: res18)')
parser.add_argument('--alpha', default=10,type=int, help='batch size')
parser.add_argument('--eta',default=10,type=int,help='rank')
parser.add_argument('--data', default='cifar40', type=str, help='in datasets: cifar40 (default: cifar40)')
parser.add_argument('--pos-label', default=0, type=int)
parser.add_argument('--data-root', default='/daintlab/data/md-ood-osr/data', type=str, help='data path')
parser.add_argument('--model-path', default='./real_final/trial-01', type=str, help='model path')
parser.add_argument('--save-path', default='./real_final/trial-01', type=str, help='save root')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu-id', default='0', type=str, help='gpu number')

args = parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cudnn.benchmark = True

    if args.model == 'res18':
        net = resnet.ResNet18(num_classes=40).cuda()

    state_dict = torch.load(f'{args.model_path}/model_200.pth')
    net.load_state_dict(state_dict)

    # evt fitting
    train_loader = dataloader.train_loader(args.data_root,args.data,args.batch_size)
    weibull_model = openmax_utils.weibull_tailfitting(net,train_loader, args.eta)

    criterion = nn.CrossEntropyLoss().cuda()
    metric_logger = utils.Logger(os.path.join(args.save_path, 'metric.log'))

    ####################################### MIS-CLASSIFICATION ####################################################

    ''' Misclassification Detection '''
    print('')
    print('Misclassification Detection')
    print('data: CIFAR40')
    print('')

    test_loader, test_targets = dataloader.test_loader(args.data_root,
                                                       args.data,
                                                       args.batch_size,
                                                       mode='test')

    # calculate score
    in_modified_score,softmax=openmax_utils.openmax(net, test_loader, weibull_model, alpha=args.alpha)
    in_modified_score=torch.tensor(in_modified_score)
    softmax=torch.tensor(softmax)

    pred_target_openmax=in_modified_score.max(axis=1)[1]
    pred_target_base = softmax.max(axis=1)[1]

    li_correct_openmax=((torch.tensor(test_targets)==pred_target_openmax)*1).tolist()
    li_correct_base=((torch.tensor(test_targets)==pred_target_base)*1).tolist()
    test_mcp=-in_modified_score[:,-1]
    acc_openmax = len(np.where(np.array(li_correct_openmax) == 1.0)[0]) / len(li_correct_openmax)
    acc_base = len(np.where(np.array(li_correct_base) == 1.0)[0]) / len(li_correct_base)
    conf_corr=sorted(zip(test_mcp, li_correct_base),key=lambda x: x[0],reverse=True)
    sorted_conf, sorted_corr = zip(*conf_corr)
    li_risk, li_cov = metrics.coverage_risk(sorted_conf, sorted_corr)

    print("acc_base : ", acc_base)
    print("acc_openmax : ", acc_openmax)
    # aurc, eaurc
    aurc, eaurc = metrics.calc_eaurc(li_risk)
    # Calibration measure ece , mce, rmsce
    ece, _, _ = metrics.calc_ece(in_modified_score,
                                 test_targets,
                                 bins=15)

    # tpr 95, aupr
    fpr_in_tpr_95, aupr, auroc = metrics.calc_fpr_aupr(in_modified_score,
                                                       li_correct_base)

    with open(f'{args.save_path}/scores.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['* Misclassification Detection Test'])
        writer.writerow(["",
                         "acc",
                         "auroc",
                         "aurc",
                         "eaurc",
                         "aupr",
                         "fpr-95%-tpr",
                         "ece"])
        writer.writerow(['', 100 * acc_openmax,
                         100 * auroc,
                         1000 * aurc,
                         1000 * eaurc,
                         100 * aupr,
                         100 * fpr_in_tpr_95,
                         100 * ece])
        writer.writerow([''])
    f.close()


####################################### OSR ############################################

    ''' Open Set Recognition '''
    ''' validation '''
    print('')
    print('Open Set Recognition/Out of Distribution Detection-Validation')
    print('known data: CIFAR40')
    print('unknown data: TinyImageNet158')
    print('')

    in_valid_loader = dataloader.in_dist_loader(args.data_root,
                                                args.data,
                                                args.batch_size,
                                                'valid')
    ood_valid_loader = dataloader.out_dist_loader(args.data_root,
                                                  'new-tinyimagenet158',
                                                  args.batch_size,
                                                  'valid')

    in_modified_score,_=openmax_utils.openmax(net, in_valid_loader, weibull_model, alpha=args.alpha)
    out_modified_score,_=openmax_utils.openmax(net, ood_valid_loader, weibull_model, alpha=args.alpha)

    in_modified_score = torch.tensor(in_modified_score)
    out_modified_score = torch.tensor(out_modified_score)

    in_valid_conf=np.asarray(-in_modified_score[:,-1])
    ood_valid_conf = np.asarray(-out_modified_score[:, -1])

    # _, _, _, in_valid_conf = utils.get_values(in_valid_loader,
    #                                           net, criterion)
    # _, _, _, ood_valid_conf = utils.get_values(ood_valid_loader,
    #                                            net, criterion)

    f1, li_f1, li_thresholds, \
    li_precision, li_recall = metrics.f1_score(in_valid_conf, ood_valid_conf,pos_label=args.pos_label)

    ood_scores = metrics.ood_metrics(in_valid_conf, ood_valid_conf)

    metric_logger.write(['osr/ood valid', '\t',
                         'det err',
                         '    auroc',
                         '      aupr-in',
                         '    aupr-out',
                         '  fpr',
                         '         f1 score', ''])
    metric_logger.write(['             ', '\t',
                         100 * ood_scores[1],
                         100 * ood_scores[2],
                         100 * ood_scores[3],
                         100 * ood_scores[4],
                         100 * ood_scores[0],
                         f1, ''])

    plot.draw_f1(args.save_path, f1, li_f1, li_thresholds,
                 data='tinyimagenet158', mode='valid', task='OsR&OoD')

    # save to .csv
    with open(f'{args.save_path}/scores.csv', 'a', newline='') as f:
        columns = ["",
                   "fpr-95%-tpr"
                   "det err",
                   "auroc",
                   "aupr-in",
                   "aupr-out",
                   "f1 score"]
        writer = csv.writer(f)
        writer.writerow(['* Open Set Recognition/Out of Distribution Detection Validation-TinyImageNet158'])
        writer.writerow(columns)
        writer.writerow(['', 100 * ood_scores[0],
                         100 * ood_scores[1],
                         100 * ood_scores[2],
                         100 * ood_scores[3],
                         100 * ood_scores[4],
                         f1])
        writer.writerow([''])
    f.close()


##################################### OSR TEST ####################################
    ''' test '''
    print('')
    print('Open Set Recognition-Test')
    print('known data: CIFAR40')
    print('unknown data: CIFAR60')
    print('')
    in_test_loader = dataloader.in_dist_loader(args.data_root,
                                               args.data,
                                               args.batch_size,
                                               'test')
    ood_test_loader = dataloader.out_dist_loader(args.data_root,
                                                 'cifar60',
                                                 args.batch_size,
                                                 'test')

    in_modified_score,_ = openmax_utils.openmax(net, in_test_loader, weibull_model, alpha=args.alpha)
    out_modified_score,_ = openmax_utils.openmax(net, ood_test_loader, weibull_model, alpha=args.alpha)

    in_modified_score = torch.tensor(in_modified_score)
    out_modified_score = torch.tensor(out_modified_score)

    in_test_conf = np.asarray(-in_modified_score[:, -1])
    ood_test_conf = np.asarray(-out_modified_score[:, -1])

    # _, _, _, in_test_conf = utils.get_values(in_test_loader,
    #                                          net,
    #                                          criterion)
    # _, _, _, ood_test_conf = utils.get_values(ood_test_loader,
    #                                           net,
    #                                           criterion)

    f1, li_f1, li_thresholds, \
    li_precision, li_recall = metrics.f1_score(in_test_conf, ood_test_conf,
                                              pos_label=args.pos_label)

    ood_scores = metrics.ood_metrics(in_test_conf, ood_test_conf)

    metric_logger.write(['osr test     ', '\t',
                         'det err',
                         '    auroc',
                         '      aupr-in',
                         '    aupr-out',
                         '  fpr',
                         '         f1 score', ''])
    metric_logger.write(['             ', '\t',
                         100 * ood_scores[1],
                         100 * ood_scores[2],
                         100 * ood_scores[3],
                         100 * ood_scores[4],
                         100 * ood_scores[0],
                         f1, ''])

    plot.draw_f1(args.save_path, f1, li_f1, li_thresholds, data='CIFAR60',
                 mode='test', task='OsR')


    with open(f'{args.save_path}/scores.csv', 'a', newline='') as f:
        columns = ["",
                   "fpr-95%-tpr",
                   "det err",
                   "auroc",
                   "aupr-in",
                   "aupr-out",
                   "f1 score"]
        writer = csv.writer(f)
        writer.writerow(['* Open Set Recognition Test-CIFAR60'])
        writer.writerow(columns)
        writer.writerow(['',
                         100 * ood_scores[0],
                         100 * ood_scores[1],
                         100 * ood_scores[2],
                         100 * ood_scores[3],
                         100 * ood_scores[4],
                         f1])
        writer.writerow([''])
    f.close()


######################################## OOD SVHN ##########################################

    ''' Out of Distribution Detection '''
    ''' test '''
    print('')
    print('Out of Distribution Detection-Test')
    print('known data: CIFAR40')
    print('unknown data: SVHN')
    print('')
    in_test_loader = dataloader.in_dist_loader(args.data_root,
                                               args.data,
                                               args.batch_size,
                                               'test')
    ood_test_loader = dataloader.out_dist_loader(args.data_root,
                                                 'svhn',
                                                 args.batch_size,
                                                 'test')

    in_modified_score,_ = openmax_utils.openmax(net, in_test_loader, weibull_model, alpha=args.alpha)
    out_modified_score,_ = openmax_utils.openmax(net, ood_test_loader, weibull_model, alpha=args.alpha)
    in_modified_score = torch.tensor(in_modified_score)
    out_modified_score = torch.tensor(out_modified_score)

    in_test_conf = np.asarray(-in_modified_score[:, -1])
    ood_test_conf = np.asarray(-out_modified_score[:, -1])

    # _, _, _, in_test_conf = utils.get_values(in_test_loader,
    #                                          net,
    #                                          criterion)
    #
    # _, _, _, ood_test_conf = utils.get_values(ood_test_loader,
    #                                           net,
    #                                           criterion)

    f1, li_f1, li_thresholds, \
    li_precision, li_recall = metrics.f1_score(in_test_conf, ood_test_conf,
                                              pos_label=args.pos_label)


    ood_scores = metrics.ood_metrics(in_test_conf, ood_test_conf)


    metric_logger.write(['ood test-svhn', '\t',
                         'det err',
                         '    auroc',
                         '      aupr-in',
                         '    aupr-out',
                         '  fpr',
                         '         f1 score', ''])

    metric_logger.write(['             ', '\t',
                         100 * ood_scores[1],
                         100 * ood_scores[2],
                         100 * ood_scores[3],
                         100 * ood_scores[4],
                         100 * ood_scores[0],
                         f1, ''])

    plot.draw_f1(args.save_path, f1, li_f1, li_thresholds, data='SVHN',
                 mode='test', task='OoD')


    with open(f'{args.save_path}/scores.csv', 'a', newline='') as f:
        columns = ["",
                   "fpr-95%-tpr",
                   "det err",
                   "auroc",
                   "aupr-in",
                   "aupr-out",
                   "f1 score"]
        writer = csv.writer(f)
        writer.writerow(['* Out of Distribution Detection Test-SVHN'])
        writer.writerow(columns)
        writer.writerow(['',
                         100 * ood_scores[0],
                         100 * ood_scores[1],
                         100 * ood_scores[2],
                         100 * ood_scores[3],
                         100 * ood_scores[4],
                         f1])
        writer.writerow([''])
    f.close()


################################ OOD LSUN ##################################
    print('')
    print('Out of Distribution Detection-Test')
    print('known data: CIFAR40')
    print('unknown data: LSUN')
    print('')
    ood_test_loader = dataloader.out_dist_loader(args.data_root,
                                                 'lsun-fix',
                                                 args.batch_size,
                                                 'test')

    # _, _, _, ood_test_conf = utils.get_values(ood_test_loader,
    #                                           net,
    #                                           criterion)


    out_modified_score,_ = openmax_utils.openmax(net, ood_test_loader, weibull_model, alpha=args.alpha)
    out_modified_score = torch.tensor(out_modified_score)

    ood_test_conf = np.asarray(-out_modified_score[:, -1])

    f1, li_f1, li_thresholds, \
    li_precision, li_recall = metrics.f1_score(in_test_conf, ood_test_conf,
                                              pos_label=args.pos_label)


    ood_results = metrics.ood_metrics(in_test_conf, ood_test_conf)


    metric_logger.write(['ood test-lsun', '\t',
                         'det err',
                         '    auroc',
                         '      aupr-in',
                         '    aupr-out',
                         '  fpr',
                         '         f1 score',
                         ''])

    metric_logger.write(['             ', '\t',
                         100 * ood_results[1],
                         100 * ood_results[2],
                         100 * ood_results[3],
                         100 * ood_results[4],
                         100 * ood_results[0],
                         f1, ''])

    plot.draw_f1(args.save_path, f1, li_f1, li_thresholds, data='LSUN',
                 mode='test', task='OoD')

    with open(f'{args.save_path}/scores.csv', 'a', newline='') as f:
        columns = ["",
                   "fpr-95%-tpr",
                   "det err",
                   "auroc",
                   "aupr-in",
                   "aupr-out",
                   "f1 score"]
        writer = csv.writer(f)
        writer.writerow(['* Out of Distribution Detection Test-LSUN'])
        writer.writerow(columns)
        writer.writerow(['',
                         100 * ood_results[0],
                         100 * ood_results[1],
                         100 * ood_results[2],
                         100 * ood_results[3],
                         100 * ood_results[4],
                         f1])
        writer.writerow([''])
    f.close()


####################################### OSR ############################################

    ''' Open Set Recognition '''
    ''' validation '''
    print('')
    print('Open Set Recognition/Out of Distribution Detection-test')
    print('known data: CIFAR40')
    print('unknown data: TinyImageNet158')
    print('')

    in_valid_loader = dataloader.in_dist_loader(args.data_root,
                                                args.data,
                                                args.batch_size,
                                                'test')
    ood_valid_loader = dataloader.out_dist_loader(args.data_root,
                                                  'new-tinyimagenet158',
                                                  args.batch_size,
                                                  'test')

    in_modified_score,_=openmax_utils.openmax(net, in_valid_loader, weibull_model, alpha=args.alpha)
    out_modified_score,_=openmax_utils.openmax(net, ood_valid_loader, weibull_model, alpha=args.alpha)

    in_modified_score = torch.tensor(in_modified_score)
    out_modified_score = torch.tensor(out_modified_score)

    in_valid_conf=np.asarray(-in_modified_score[:,-1])
    ood_valid_conf = np.asarray(-out_modified_score[:, -1])

    # _, _, _, in_valid_conf = utils.get_values(in_valid_loader,
    #                                           net, criterion)
    # _, _, _, ood_valid_conf = utils.get_values(ood_valid_loader,
    #                                            net, criterion)

    f1, li_f1, li_thresholds, \
    li_precision, li_recall = metrics.f1_score(in_valid_conf, ood_valid_conf,pos_label=args.pos_label)

    ood_scores = metrics.ood_metrics(in_valid_conf, ood_valid_conf)

    metric_logger.write(['osr/ood valid', '\t',
                         'det err',
                         '    auroc',
                         '      aupr-in',
                         '    aupr-out',
                         '  fpr',
                         '         f1 score', ''])
    metric_logger.write(['             ', '\t',
                         100 * ood_scores[1],
                         100 * ood_scores[2],
                         100 * ood_scores[3],
                         100 * ood_scores[4],
                         100 * ood_scores[0],
                         f1, ''])

    plot.draw_f1(args.save_path, f1, li_f1, li_thresholds,
                 data='tinyimagenet158', mode='valid', task='OsR&OoD')

    # save to .csv
    with open(f'{args.save_path}/scores.csv', 'a', newline='') as f:
        columns = ["",
                   "fpr-95%-tpr"
                   "det err",
                   "auroc",
                   "aupr-in",
                   "aupr-out",
                   "f1 score"]
        writer = csv.writer(f)
        writer.writerow(['* Open Set Recognition/Out of Distribution Detection test-TinyImageNet158'])
        writer.writerow(columns)
        writer.writerow(['', 100 * ood_scores[0],
                         100 * ood_scores[1],
                         100 * ood_scores[2],
                         100 * ood_scores[3],
                         100 * ood_scores[4],
                         f1])
        writer.writerow([''])
    f.close()


if __name__ == "__main__":
    main()
