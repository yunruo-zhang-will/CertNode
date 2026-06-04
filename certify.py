import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm, trange
from scipy.stats import norm, binomtest
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import ncx2
from utils import *


def certify(rho, top2, count1, count2, sample_config, nclass, args):
    """
    parameters:
    rho: int, number of node perturbations to certify against
    top2: np array of shape [num_nodes, 2], the top 2 predicted classes for each node
    count1: list of length num_nodes, the counts of the top predicted class for each node
    count2: list of length num_nodes, the counts of the second top predicted class for each node
    sample_config: dict, contains 'p_e' and 'p_n'
    nclass: int, number of classes
    args: arguments, contains 'degree_budget', 'n_smoothing', 'conf_alpha', 'singleton'

    return: 
    cAHat_list: list of length num_nodes, the predicted class if certified, -1 otherwise
    certified_list: list of length num_nodes, True if certified, False otherwise
    """
    cAHat_list = []
    certified_list = []
    p_n = sample_config['p_n']
    for i in range(len(count1)):
        yAHat = top2[i, 0]
        p_tilde = np.power(p_n, rho)

        pABar = proportion_confint(count1[i], args.n_smoothing, alpha=2 * args.conf_alpha / nclass, method="beta")[0]
        pBBar = proportion_confint(count2[i], args.n_smoothing, alpha=2 * args.conf_alpha / nclass, method="beta")[1]
        certified = (p_tilde * (pABar - pBBar + 1) > 1)

        #print(f'count1[{i}]: {count1[i]}, count2[{i}]: {count2[i]}')
        s = binomtest(count1[i], count1[i] + count2[i], p=0.5)
        if s.pvalue < args.conf_alpha:
            cAHat_list.append(yAHat)
            certified_list.append(certified)
        else:
            cAHat_list.append(-1)
            certified_list.append(False)
            continue
    return cAHat_list, certified_list


def get_pA_pB(top2, count1, count2, sample_config, nclass, args, total_votes=None):
    pA_list = []
    pB_list = []
    yA_list = []
    for i in range(len(count1)):
        yAHat = top2[i, 0]

        pABar = proportion_confint(count1[i], args.n_smoothing, alpha=2 * args.conf_alpha / nclass, method="beta")[0]
        pBBar = proportion_confint(count2[i], args.n_smoothing, alpha=2 * args.conf_alpha / nclass, method="beta")[1]

        yA_list.append(yAHat)
        pA_list.append(pABar)
        pB_list.append(pBBar)
    return yA_list, pA_list, pB_list


