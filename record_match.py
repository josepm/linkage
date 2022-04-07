"""
__author__: josep ferrandiz

    dfL, dfR DFs to link records (one in each row)
    - cols to score have the same name if dfL and dfR
    - find at most one match for each dfL row in dfR
    - support for multiple levels of matching in each column
    - assumes imputation (if any) and data clean up, data features (sound) already completed. See preprocessing.py
    - scores are implemented in the file scores.py

    m_dict = {..., key: [match_op, match_levels, cols], ...}
        match_op: the name of the scoring function
        op_levels: can be one of,
            a nbr (nbr of levels), -> build a list of equally spaced values, eg op_levels = 3 would result in [0, 0.5, 1.0]
            list of thresholds: [0 = v_0, v_1, ..., v_{L-1} = 1]
            Note: meaning of scores:
                    0: different
                    1: identical
                    in between levels: similar, the higher the more similar
        cols: the name of the colsused as arguments in the scoring function
    restrict dfL and dfR to scoring cols and drop duplicates
    checks dfL, dfR have usable (i.e., no duplicated values) indices
    use case dependency:
        scoring functions are use case dependent
        the class Compare is use case dependent and needs to be updated with scoring function names (over time this will be complete enough)


Processing steps (assumes scoring functions and Compare() class are updated to the use case)
    define the matching dict, m_dict
    add_features(dfL, dfR, cols)                                # feature eng (sound, ...) TODO
    preprocess(dfL, dfR)                                        # clean data: string, mail, zip, phone, ... TODO
    m_obj = RecordMatchModel(dfA, ixA, dfB, ixB, m_dict)
    m_obj.set_index()      # joint index (cartesian)
    m_obj.set_scores()     # observed gammas and deltas
    m_obj.em()
    kwargs = {...}         # set obj_func params and model args
    match_df = m_obj.record_matches(obj_func, kwargs)
"""

# TODO: implement data clean up functions (strings, phones, zips, ...) -- see validation.py,  preprocessing and record_linkgae package
# TODO: feature eng (sound, ...) -- see record_linkage package
# TODO: use dask to support larger DFs


import sys
import os
import numpy as np
import pandas as pd
import logging
import time
import my_utilities.utilities as ut
import sales_match.record_linkage.score as score
import copy
from scipy.optimize import minimize_scalar
from functools import partial


log_format = '%(asctime)s - %(process)d - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')        # To override the default severity of logging
formatter = logging.Formatter(log_format, "%Y-%m-%d %H:%M:%S")
formatter.converter = time.gmtime

if not logger.hasHandlers():
    file_handler = logging.StreamHandler(sys.stdout)  # file_handler = logging.FileHandler("mylogs_model.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class RecordMatchModel(object):
    def __init__(self, dfL, idxL, dfR, idxR, dict_cols, NA=None, max_iter=1000):
        """
        matches left DF data to right DF data so that the entries in left DF are matched at most once.
        :param dfL: DF with records to match
        :param dfR: DF with records to match
        :param idxL: index col for dfL
        :param idxR: index col for dfR
        :param NA: if None treat missing values as a mismatch (i.e. score=0 when at least one of the L or R values is missing -NA, NULL, ...),
                   else ignore scores (i.e. drop entry altogether) when at least one of the L or R values is missing
        :param max_iter: max EM iterations
        :param dict_cols: a dict with info on the cols to merge as keys and the score op details on {..., col: [score_op, op_levels, dtype], ...}
        """
        self.dict_cols = dict_cols
        self.dfL, self.ixL = self.check_df(dfL, idxL)
        self.dfR, self.ixR = self.check_df(dfR, idxR)
        self.dtypesL = {k: str(v) for k, v in self.dfL.dtypes.to_dict().items()}
        self.dtypesR = {k: str(v) for k, v in self.dfR.dtypes.to_dict().items()}
        self.max_iter = max_iter
        self.em_ctr = 0
        self.min_tol = 1.0E-06
        self.max_match = min(len(self.dfL), len(self.dfR)) / (len(self.dfL) * len(self.dfR))
        self.em_done = False
        self.NA = NA

        ut.set_data_types(self.dfL, data_types=self.dtypesL, verbose=True, by_date=False)
        ut.set_data_types(self.dfR, data_types=self.dtypesR, verbose=True, by_date=False)

    # ###################################
    # set up
    # ###################################
    def check_df(self, a_df, a_idx):
        # df.index is not a_idx, a_idx should be a column in a_df
        # check DFs: type, idx is a col, m_cols, dups on m_cols
        if not(isinstance(a_df, pd.core.frame.DataFrame)):
            logger.error('invalid input type. Should be a DF')
            return None, None
        else:
            if not(a_idx in a_df.columns):
                logger.error('invalid index: ' + str(a_idx) + ' only single column indices implemented')
                return None, None
            if a_idx in self.dict_cols.keys():
                logger.error(str(a_idx) + ' index cannot be a score column')
                return None, None

            to_pop = list()
            score_cols = list()
            for k, vlist in self.dict_cols.items():
                for c in vlist[2]:
                    if c not in a_df.columns:
                        to_pop.append(k)
                    else:
                        score_cols.append(c)

            for c in to_pop:
                logger.warning('dropping match key ' + str(c) + ': not in DF')
                self.dict_cols.pop(c)

            df = a_df[[a_idx] + score_cols].copy()
            df.dropna(subset=[a_idx], inplace=True)
            df.drop_duplicates(inplace=True)
            return df, a_idx

    def set_index(self):
        # cross-merges the left and right DFs
        # target indices must be in the columns
        # this seems faster than a direct cross-merge
        if self.ixL not in self.dfL.columns:
            logger.error(self.ixL + ' must be a column')
            return None
        else:
            self.dfL.reset_index(inplace=True, drop=True)
        if self.ixR not in self.dfR.columns:
            logger.error(self.ixR + ' must be a column')
            return None
        else:
            self.dfR.reset_index(inplace=True, drop=True)

        start = time.time()
        self.m_df = self.dfL.merge(self.dfR, how='cross', suffixes=('_L', '_R'))
        logger.info('index set in ' + str(np.round(time.time()-start, 2)) + 'secs')
        return None

    def set_scores(self):
        start = time.time()
        for k, clist in self.dict_cols.items():
            not_nulls, gamma = self.set_scores_(k, clist)
            self.m_df[k + '_gamma'] = np.array(gamma, dtype=pd.Float32Dtype)            # gets converted to NA support later
            self.m_df[k + '_delta'] = np.array(not_nulls, dtype=pd.BooleanDtype)
            self.m_df.set_index(k + '_delta', inplace=True)
            self.m_df.loc[self.m_df.index == True, k + '_gamma'] = np.nan               # Must be np.nan here. pd.NA support conversion is later
            self.m_df.reset_index(inplace=True)
        self.m_df = self.m_df.convert_dtypes()
        self.d_cols = [c for c in self.m_df.columns if '_delta' in c]
        self.g_cols = [c for c in self.m_df.columns if '_gamma' in c]
        d_df = self.m_df[self.d_cols + self.g_cols].copy()

        # when self.NA != None: normalize the _delta columns to avoid favoring records with many missing values
        # delta_den = len(self.d_cols) - d_df[self.d_cols].sum(axis=1)
        # d_df = d_df[delta_den > 0].copy()   # drop record pairs with missing data in all scores
        # for c in self.d_cols:
        #     d_df[c] /= delta_den

        # this groupby will drop all the NA entries so that delta is not needed in the EM computation
        self.d_df = pd.DataFrame(d_df.groupby(self.d_cols + self.g_cols).size(), columns=['count'])  # compute gamma's for unique combinations only
        self.d_df.reset_index(inplace=True)
        self.d_df = self.d_df.convert_dtypes()

        # create the values arrays based on the actually observed gamma values (this avoids 0 values in nus and mus)
        self.values = {c: list(self.d_df[self.d_df[c + '_delta'] == 1][c + '_gamma'].unique()) for c in self.dict_cols.keys()}
        for c, vals in self.values.items():
            if len(vals) <= 1:
                self.values.pop(c)
                logger.warning('dropping column ' + str(c) + ': not enough unique data')

        logger.info('scores set in ' + str(np.round(time.time()-start, 2)))

        # drop delta columns (not needed any longer)
        self.m_df.drop(self.d_cols, axis=1, inplace=True)
        self.d_df.drop(self.d_cols, axis=1, inplace=True)

    def set_scores_(self, k, clist):
        cmp_op, levels, cols = clist
        cmp_obj = Compare(k, cmp_op, levels)
        l_cols = [c + '_L' for c in cols]
        r_cols = [c + '_R' for c in cols]

        # switch to series if only one entry
        if len(l_cols) == 1:
            l_cols = l_cols[0]
        if len(r_cols) == 1:
            r_cols = r_cols[0]

        start = time.time()
        nulls, gamma = cmp_obj.cmp_score(self.m_df[l_cols], self.m_df[r_cols])  # nulls = missing data boolean, gamma = scores
        nulls = np.nan_to_num(nulls, nan=False)
        nulls = nulls.astype('bool')
        logger.info('scored key ' + str(k) + ' in ' + str(np.round(time.time() - start, 2)) + 'secs')
        if self.NA is None:                         # set missing values to no-match
            gamma[nulls] = 0                        # 0 score for missing values
            nulls = np.array([False] * len(gamma))  # ignore missing values
        return ~nulls, gamma

    # ###############################
    # model estimation
    # ###############################
    def em_init(self):
        # initialize EM algorithm
        self.lmbda = np.random.uniform(low=0.0, high=self.max_match)
        self.mu_hat, self.nu_hat = dict(), dict()
        self.b_gamma = dict()
        for c in list(self.dict_cols.keys()):
            self.mu_hat[c] = self.em_init_col(c, scale=1.0)
            self.nu_hat[c] = self.em_init_col(c, scale=10.0)  # p(match|no match) should be smaller than p(match|match)
            self.b_gamma[c] = dict()
            self.b_gamma[c] = {ix: np.array(self.d_df[c + '_gamma'].values == v).astype(np.int8) for ix, v in enumerate(self.values[c])}

    def em_init_col(self, c, scale=1.0):
        p = np.random.uniform(low=0.0, high=self.max_match / scale)
        if len(self.values[c]) == 2:
            return np.array([1 - p, p])
        else:  # Note: do not init proportional to gamma counts: it converges very slowly if at all
            arr = np.random.uniform(low=0.0, high=1.0 - p, size=len(self.values[c]) - 1)
            arr *= ((1 - p) / np.sum(arr))
            return np.array(list(arr) + [p])

    def em_e(self):
        # EM E-step
        l_mus = np.zeros(len(self.d_df))                               # len = record pairs = len(d_df)
        l_nus = np.zeros(len(self.d_df))                               # len = record pairs = len(d_df)
        for c in list(self.dict_cols.keys()):
            for ix, v in enumerate(self.values[c]):
                l_mus += self.lmp(self.mu_hat[c][ix], self.b_gamma[c][ix])
                l_nus += self.lmp(self.nu_hat[c][ix], self.b_gamma[c][ix])
        lw_i = (l_mus - l_nus)                                                   # agreement odds (log)
        mus = np.exp(l_mus)
        nus = np.exp(l_nus)
        lwbar_i = np.log((1 - mus) / (1 - nus))                                  # disagreement odds (log)
        xi_i = 1.0 / (1.0 + ((1.0 - self.lmbda) / self.lmbda) * np.exp(-lw_i))   # match prob | gammas
        return xi_i, lw_i, lwbar_i, mus, nus

    @staticmethod
    def lmp(x, p):
        with ut.suppress_stdout_stderr():   # suppress log(0) warning
            v = p * np.log(x)               # NAs when p = 0 and x = 0, p = 0 and x = np.inf
            s = pd.Series(v).fillna(0.0)
            return s.values

    def em_m(self):
        # EM M-step
        mu_den = self.xi * self.d_df['count'].values
        mu_den_sum = np.sum(mu_den)
        nu_den = (1 - self.xi) * self.d_df['count'].values
        nu_den_sum = np.sum(nu_den)

        self.lmbda = mu_den_sum / self.d_df['count'].sum()
        if self.lmbda > self.max_match:
            logger.warning('lambda exceeded upper bound: ' + str(self.lmbda) + ' > ' + str(self.max_match))
            self.lmbda = self.max_match / 2.0

        for c in list(self.dict_cols.keys()):
            for ix, v in enumerate(self.values[c]):
                # mu_hat^c_ix
                num = mu_den * self.b_gamma[c][ix]
                self.mu_hat[c][ix] = np.sum(num) / mu_den_sum  # ix: index in the values array

                # ensure sum_ix mu_hat^c_ix = 1
                mu_sum = np.sum(self.mu_hat[c])
                self.mu_hat[c] /= mu_sum

                # nu_hat^c_ix
                num = nu_den * self.b_gamma[c][ix]
                self.nu_hat[c][ix] = np.sum(num) / nu_den_sum

                # ensure sum_ix nu_hat^c_ix = 1
                nu_sum = np.sum(self.nu_hat[c])
                self.nu_hat[c] /= nu_sum

    def em(self):           # EM loop
        if self.em_ctr > 10:
            logger.error('EM did not converge: ' + str(self.em_ctr))
            return -1
        else:
            n_iter, tol = 0, np.inf
            self.em_init()
            wbari, m_hat, u_hat = None, None, None
            while n_iter < self.max_iter and tol > self.min_tol:
                mu_old = copy.deepcopy(self.mu_hat)
                nu_old = copy.deepcopy(self.nu_hat)
                lmbda_old = self.lmbda

                self.xi, self.wi, wbari, m_hat, u_hat = self.em_e()
                if self.xi is None:
                    self.em_ctr += 1
                    return self.em()
                self.em_m()

                n_iter += 1
                l_err = np.abs(1 - self.lmbda / lmbda_old)
                tol = max(l_err, self.d_tol(mu_old, self.mu_hat),  self.d_tol(nu_old, self.nu_hat))

            self.d_df['xi'] = self.xi
            self.d_df['w'] = self.wi     # log(agreement odds)
            self.d_df['w_bar'] = wbari   # log(disagreement odds)
            self.d_df['m_hat'] = m_hat
            self.d_df['u_hat'] = u_hat
            self.em_done = True

            string = '\nEM Results\n'
            for k, v in self.mu_hat.items():
                vv = [np.round(x, 6) for x in v]
                ww = [np.round(x, 6) for x in self.nu_hat[k]]
                string += 'em_ctr: ' + str(self.em_ctr) + ' iter: ' + str(n_iter) + ' tol: ' + str(np.round(tol, 8)) + \
                          ' col: ' + k + ' lmbda: ' + str(np.round(self.lmbda, 6)) + ' mu: ' + str(vv) + ' nu: ' + str(ww) + '\n'
            logger.info(string)
            return 1

    @staticmethod
    def d_tol(d1, d2):  # tolerance
        tol = 0.0
        for k, v1 in d1.items():
            v2 = d2[k]
            if np.min(v1) != 0.0:
                tol += np.sqrt(np.mean((1 - v2 / v1) ** 2))
            else:
                i1 = np.argmin(v1)
                if v2[i1] != 0.0:  # if v2[i1] == 0, all OK: we converged: Do not increase tol. Otherwise, bad (add 1 to tol)
                    tol += 1.0
                    logger.warning('??????')
        return tol / len(d1.keys())

    # ###############################
    # record matching
    # ###############################
    def record_matches(self, obj_func, *fpars):
        # find xi-threshold that minimizes obj_func, xi = prob(match)
        args = tuple([self] + list(fpars))
        res = minimize_scalar(obj_func, args=args, bounds=(0.0, 1.0), method='bounded')
        thres = res.x if res.success == True else None
        return None if thres is None else self.ll_matches_(min_xi=thres)

    def ll_matches_(self, min_xi=None):
        # finds the best match for each left index in ixL
        if min_xi is None:
            logger.warning('min_xi not set. Using 0')
            min_xi = 0.0
        self.d_df['l_odds_diff'] = self.d_df['w'] - self.d_df['w_bar']
        d_df = self.d_df[(self.d_df['l_odds_diff'] > 0.0) & (self.d_df['xi'] >= min_xi)]
        ll_df = self.m_df.merge(d_df, on=self.g_cols, how='left')
        ll_df.dropna(subset=['l_odds_diff', 'xi'], inplace=True)
        return ll_df.groupby(self.ixL).apply(lambda f: f[f['l_odds_diff'] == f['l_odds_diff'].max()].sample(n=1, axis=0)).reset_index(drop=True)

    # ###############################################
    # utility functions to compute optimal threshold
    # ###############################################
    def pos_(self, s):  # FP + TP
        v = np.where(self.xi >= s, 1, 0)
        p = v * self.d_df['count'].values
        fp = p * (1.0 - self.xi)   # FP
        tp = p * self.xi
        return p, fp, tp

    def neg_(self, s):  # FN + TN
        v = np.where(self.xi < s, 1, 0)
        n = v * self.d_df['count'].values
        fn = n * self.xi          # FN
        tn = n * (1 - self.xi)
        return n, fn, tn

    def FDR(self, s):   # FDR = FP / (FP + TP)  False discovery rate
        p, fp, tp = self.pos_(s)
        try:
            return np.sum(fp) / np.sum(p)
        except ZeroDivisionError:
            logger.error('invalid FDR for s ' + str(s))
            return np.nan

    def FNR(self, s):  # FNR = FN / (FN + TP)   Miss
        n, fn, tn = self.neg_(s)
        try:
            return np.sum(fn) / np.sum(n)
        except ZeroDivisionError:
            logger.error('invalid FNR for s ' + str(s))
            return np.nan

    def FPR(self, s):  # FPR = FP / (FP + TN)   False alarm
        p, fp, tp = self.pos_(s)
        n, fn, tn = self.neg_(s)
        try:
            return np.sum(fp) / (np.sum(fp) + np.sum(tn))
        except ZeroDivisionError:
            logger.error('invalid FPR for s ' + str(s))
            return np.nan

    def Precision(self, s):   # TP / (TP + FP)
        p, fp, tp = self.pos_(s)
        try:
            return np.sum(tp) / (np.sum(tp) + np.sum(fp))
        except ZeroDivisionError:
            if s == 0.0:
                return 1.0
            elif s == 1.0:
                return 0.0
            else:
                logger.error('invalid precision for s ' + str(s))
                return np.nan

    def Recall(self, s):      # TP / (TP + FN)
        p, fp, tp = self.pos_(s)
        n, fn, tn = self.neg_(s)
        try:
            return np.sum(tp) / (np.sum(tp) + np.sum(fn))
        except ZeroDivisionError:
            if s == 0.0:
                return 1.0
            elif s == 1.0:
                return 0.0
            else:
                logger.error('invalid recall for s ' + str(s))
                return np.nan


class Compare(object):
    def __init__(self, name, op_name, op_levels):
        # name: could be the col name
        # cmp_op(x, y) scores column x vs column y
        # op_levels: None (defaults to 2), a nbr (nbr of levels, equally spaced), list of thresholds: [0 = v_0, v_1, ..., v_{L-1} = 1]
        #       0: different
        #       1: identical
        #       in between levels: similar, equally spaced between 0 and 1 (default) but thresholds can be set
        # op_levels as int includes 0 and 1 always
        self.name = name
        self.op_name = op_name
        if isinstance(op_levels, type(None)):
            self.op_levels = 2
            self.values = [0, 1]
        elif isinstance(op_levels, list) or isinstance(op_levels, tuple):
            self.values = np.sort(np.array(list(set(op_levels))))
        elif isinstance(int(op_levels), int):
            self.values = np.linspace(0.0, 1.0, max(2, op_levels))
        else:
            logger.error('invalid op_levels: ' + str(op_levels))
        self.set_cmp_op(op_name)

    def cmp_score(self, v1, v2):
        if self.cmp_op is None:
            return None
        else:
            g = self.cmp_op(v1, v2)  # nulls, scores
            if self.op_name == 'exact':
                return g
            elif self.op_name == 'string':
                ix = np.digitize(g[1], self.values) - 1
                vals = self.values[ix]
                return g[0], vals
            elif self.op_name == 'cityzip':
                return g
            else:
                logger.error(str(self.op_name) + ': not implemented')
                return None, None

    def set_cmp_op(self, op_name):
        if op_name == 'exact':
            self.cmp_op = score.cmp_exact
        elif op_name == 'string':
            self.cmp_op = score.cmp_string
        elif op_name == 'cityzip':
            self.cmp_op = score.cmp_cityzip
        else:
            logger.error(str(op_name) + ' not implemented')
            self.cmp_op = None

