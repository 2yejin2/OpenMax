from __future__ import print_function
import torch
import numpy as np
from sklearn import metrics

import utils

def get_curve(in_output, ood_output, stypes=['Baseline']):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()
    for stype in stypes:
        known = in_output
        novel = ood_output

        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1:] = tp[stype][l]
                fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        fpr_at_tpr95[stype] = fp[stype][tpr95_pos] / num_n

    return tp, fp, fpr_at_tpr95


def ood_metrics(in_output, ood_output, stypes=['result'], verbose=False):
    tp, fp, fpr_at_tpr95 = get_curve(in_output, ood_output, stypes)
    results = dict()

    mtypes = ['FPR95', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')

    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results = dict()

        # TNR
        mtype = 'FPR95'
        results[mtype] = fpr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
        results[mtype] = -np.trapz(1. - fpr, tpr)
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

        # DTERR
        mtype = 'DTERR'
        results[mtype] = 1 - (.5 * (tp[stype] / tp[stype][0] + 1. - fp[stype] / fp[stype][0]).max())
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

        # AUIN
        mtype = 'AUIN'
        denom = tp[stype] + fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype] / denom, [0.]])
        results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0] - tp[stype] + fp[stype][0] - fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0] - fp[stype]) / denom, [.5]])
        results[mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[mtype][mtype]), end='')
            print('')

    mtypes = ['FPR95', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100. * results['FPR95']), end='')
    print(' {val:6.2f}'.format(val=100. * results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100. * results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100. * results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100. * results['AUOUT']), end='')
    print('')

    fpr = 100. * results['FPR95']
    det_err = 100. * results['DTERR']
    auroc = 100. * results['AUROC']
    aupr_in = 100. * results['AUIN']
    aupr_out = 100. * results['AUOUT']

    return fpr, det_err, auroc, aupr_in, aupr_out


def f1_score(in_conf, out_conf, pos_label):
    in_conf = np.array(in_conf)
    in_label = np.zeros(len(in_conf))

    out_conf = np.array(out_conf)
    out_label = np.ones(len(out_conf))

    conf = np.append(in_conf, out_conf)
    labels = np.append(in_label, out_label)

    if conf.ndim != 1:
        conf_max = np.max(conf, 1)
    else:
        conf_max = conf

    conf_labels = sorted(zip(conf_max, labels),
                          key=lambda x: x[0], reverse=False)
    sorted_conf, sorted_labels = zip(*conf_labels)
    precision, recall, thresholds = metrics.precision_recall_curve(sorted_labels, sorted_conf, pos_label=pos_label)

    idx_thr_05 = np.argmin(np.abs(thresholds - 0.5))
    f1_score = 2 * (precision[idx_thr_05] * recall[idx_thr_05]) / (precision[idx_thr_05] + recall[idx_thr_05])
    li_f1_score = 2 * (precision * recall) / (precision + recall)
    thresholds = np.append(thresholds, 1.0)

    return f1_score, li_f1_score, thresholds, precision, recall



# AUPR ERROR
def calc_fpr_aupr(softmax, correct):
    softmax = np.array(softmax)
    correct = np.array(correct)

    if softmax.ndim != 1:
        softmax_max = np.max(softmax, 1)
    else:
        softmax_max = softmax

    fpr, tpr, thresholds = metrics.roc_curve(correct, softmax_max)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95_tpr = fpr[idx_tpr_95]

    auroc = metrics.roc_auc_score(correct, softmax_max)

    aupr_err = metrics.average_precision_score(-1 * correct + 1, -1 * softmax_max)

    print('* FPR@95%TPR: ', fpr_at_95_tpr)
    print('* AUPR-ERR: ', aupr_err)
    print('* AUROC: ', auroc)

    return fpr_at_95_tpr, aupr_err, auroc



# calc ece
def calc_ece(softmax, label, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)

    confidence, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels)

    ece = torch.zeros(1)

    li_acc = []
    li_count = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidence.gt(bin_lower.item()) * confidence.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = confidence[in_bin].mean()
            li_count.append(len(correctness[in_bin]))

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        else:
            li_count.append(0)
            accuracy_in_bin = 0.0

        li_acc.append(accuracy_in_bin)

    print('* ECE: {0:.6f}'.format(ece.item()))

    return ece.item(), li_acc, li_count



# coverage , risk
def coverage_risk(rank_confidence, rank_acc):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(rank_confidence)):
        coverage = (i + 1) / len(rank_confidence)
        coverage_list.append(coverage)

        if rank_acc[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list



# E-AURC calculation
def calc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    AURC = risk_coverage_curve_area
    EAURC = risk_coverage_curve_area - optimal_risk_area

    print('* AURC: {0:.6f}'.format(risk_coverage_curve_area))
    print('* E-AURC: {0:.6f}'.format(EAURC))

    return AURC, EAURC


def md_metrics(test_loader, test_label, net, criterion):

    softmax, correct, logit, mcp = utils.get_values(test_loader, net, criterion)
    conf_corr = sorted(zip(mcp, correct), key=lambda x: x[0], reverse=True)
    sorted_conf, sorted_corr = zip(*conf_corr)
    li_risk, li_cov = coverage_risk(sorted_conf, sorted_corr)

    acc = len(np.where(np.array(correct)==1.0)[0])/len(correct)

    # aurc, eaurc
    aurc, eaurc = calc_eaurc(li_risk)

    # Calibration measure ece , mce, rmsce
    ece, _, _ = calc_ece(softmax, test_label, bins=15)
    _, li_acc, li_count = calc_ece(softmax, test_label, bins=10)

    # tpr 95, aupr
    fpr_in_tpr_95, aupr, auroc = calc_fpr_aupr(softmax, correct)

    return acc, auroc, aurc, eaurc, fpr_in_tpr_95, aupr, ece, li_acc, li_count
