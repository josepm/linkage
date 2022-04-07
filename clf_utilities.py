"""
__author__: josep ferrandiz
classifier utility functions
this file contains:
1) objective functions for matching (examples):
    - minFR: joint minimization of FNR and FPR
    - minFBeta: F-score(beta): weighted harmonic mean of precision and recall
    - see other approaches: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5147524/

2) objective function signature:
    obj_func(s, pars, em_mdl)
    - classification threshold: s
    - parameters:
        - obj func pars like beta, q, ...
        - the record linkage model object is param
"""

import sys
import numpy as np
import logging
import time


log_format = '%(asctime)s - %(process)d - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')        # To override the default severity of logging
formatter = logging.Formatter(log_format, "%Y-%m-%d %H:%M:%S")
formatter.converter = time.gmtime

if not logger.hasHandlers():
    file_handler = logging.StreamHandler(sys.stdout)  # file_handler = logging.FileHandler("mylogs_model.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def minFbeta(s, rm_mdl, beta=None):
    return -Fbeta(s, rm_mdl, beta=beta)


def Fbeta(s, rm_mdl, beta=None):
    """
    weighted harmonic mean of precision and recall
    measures the effectiveness of retrieval with respect to a user
    who attaches beta times as much importance to recall than precision
    beta = 2: weights recall higher
    beta = 0.5: weights precision higher
    :param s: classification threshold
    :param beta: weighting parameter
    :param rm_mdl: record matching model object to the optimized
    :return: f-score(beta)
    """
    if beta is None:
        beta = 1.0
    p = rm_mdl.Precision(s)
    r = rm_mdl.Recall(s)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def minFR(s, rm_mdl, q=None):
    """
    joint minimization of FNR and FPR
    :param s: classification threshold
    :param q: relative cost of a False Positive -false alarm- (relative to the cost of a False Negative -miss-)
              q = 0 and q = inf do not make a lot of sense
    :param rm_mdl: record matching model object to the optimized
    :return: (weighted) distance to the origin
    """
    if q is None:
        q = 1.0
    if np.isinf(q) or q == 0.0:
        logger.warning('minFR has invalid cost: ' + str(q))
        return None
    else:
        return q * rm_mdl.FPR(s) ** 2 + rm_mdl.FNR(s) ** 2


def minPR(s, rm_mdl, q=None):
    """
    finds the point where TPR = 1 - FPR. Game theory based.
    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5147524/
    :param s: classification threshold
    :param q: not used. Here to maintain API
    :param rm_mdl: record matching model object to the optimized
    :return: (weighted) distance to the origin
    """
    if q is None:
        q = 1.0
    return (1 - rm_mdl.FPR(s)) ** 2 + rm_mdl.TPR(s) ** 2

