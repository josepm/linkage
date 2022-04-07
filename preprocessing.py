"""
__author__: josep ferrandiz

"""

import pandas as pd
import sys

# import pandas as pd
from recordlinkage.preprocessing import clean, phonenumbers, phonetic
from recordlinkage.base import BaseCompareFeature
import time
from joblib import Parallel, delayed
import my_utilities.utilities as ut

N_JOBS = -1

from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode
from record_linkage.record_linkage import validation as val


def data_process(f, ref_df, thres=0.95):
    print('set types to string')
    f.fillna(pd.NA, inplace=True)
    for c in f.columns:
        f[c] = f[c].astype('string[pyarrow]')

    # clean email
    start = time.time()
    with ut.suppress_stdout_stderr():
        f = set_email(f.copy())
        f = email_match(f.copy(), ref_df, thres=thres)
    print('email done in ' + str(time.time() - start) + 'secs')

    # clean columns
    start = time.time()
    with ut.suppress_stdout_stderr():
        results_list = Parallel(n_jobs=N_JOBS)(delayed(clean_string)(f[c]) for c in ['first_name', 'last_name', 'city', 'state', 'dpid'])
        for ix, c in enumerate(['first_name', 'last_name', 'city', 'state', 'dpid']):
            f[c + '_clean'] = results_list[ix]
    print('cols cleaned in ' + str(time.time() - start) + 'secs')

    # add sound
    # start = time.time()
    # with ut.suppress_stdout_stderr():
    #     results_list = Parallel(n_jobs=N_JOBS)(delayed(phonetic)(f[c + '_clean'], 'soundex', decode_error='ignore') for c in ['first_name', 'last_name', 'city'])
    #     for ix, c in enumerate(['first_name', 'last_name', 'city']):
    #         f[c + '_sound'] = results_list[ix]
    # print('sound done in ' + str(time.time() - start) + 'ses')

    # clean zip (list of valid zips???)
    # - validate zip against official zips?
    # - validate zip and city?
    # - impute city from zip?
    start = time.time()
    with ut.suppress_stdout_stderr():
        f['zip'] = zip_validate(f['zip'].copy())
    print('zip done in ' + str(time.time() - start) + 'secs')

    # clean phone numbers
    # - validate area codes?
    # - number length validation?
    # assume US only phone (len=10)
    start = time.time()
    for c in ['phone1', 'phone2', 'phone3']:
        f[c] = f[c].str.split('.', expand=True)[0].astype('string[pyarrow]')
        f[c] = f[c].str.replace('[^0-9+]+', '', regex=False).astype('string[pyarrow]')
        zlen = f[c].str.len()
        b = zlen == 10
        b.fillna(False, inplace=True)
        f[c] = pd.concat([f[c][b], pd.Series(pd.NA, index=f[c].index[~b])], axis=0).sort_index()
    print('phones done in ' + str(time.time() - start) + 'secs')

    # clean state values (valid 2-letter codes?)
    # validate zip against state?

    print('set final data types')
    for c in f.columns:
        f[c] = f[c].astype('string[pyarrow]')
    f.drop_duplicates().reset_index(inplace=True, drop=True)
    return f


def set_email(f):
    f['email'] = set_at(f['email'].copy())
    f[['email_name', 'email_domain']] = f['email'].str.split('@', expand=True, n=1)
    return f


def set_at(s_):
    n = s_.isnull()
    sn = s_[n]
    sy = s_[~n]
    s_at = sy.str.count('@')
    b_ = s_at == 1
    n = pd.Series([pd.NA] * len(sy), index=sy.index)
    return pd.concat([sy[b_], n[~b_], sn], axis=0)


def clean_string(s, lowercase=True, replace_by_none=r'[^ \-\_A-Za-z0-9]+',
                 replace_by_whitespace=r'[\-\_]', replace_whitespace=None,
                 strip_accents=None, remove_brackets=True, remove_connectors=True,
                 encoding='utf-8', decode_error='strict'):
    """Clean string variables.
    Clean strings in the Series by removing unwanted tokens,
    whitespace and brackets.
    Parameters
    ----------
    s : pandas.Series
        A Series to clean.
    lowercase : bool, optional
        Convert strings in the Series to lowercase. Default True.
    replace_by_none : str, optional
        The matches of this regular expression are replaced by ''.
    replace_by_whitespace : str, optional
        The matches of this regular expression are replaced by a
        whitespace.
    replace_whitespace : str, optional
        Replaces whitespaces by the str value if not None
    remove_brackets : bool, optional
        Remove all content between brackets and the bracket
        themselves. Default True.
    remove_connectors : bool, optional
        Remove connector words. Default True.
    strip_accents : {'ascii', 'unicode', None}, optional
        Remove accents during the preprocessing step. 'ascii' is a
        fast method that only works on characters that have an direct
        ASCII mapping. 'unicode' is a slightly slower method that
        works on any characters. None (default) does nothing.
    encoding : str, optional
        If bytes are given, this encoding is used to decode. Default
        is 'utf-8'.
    decode_error : {'strict', 'ignore', 'replace'}, optional
        Instruction on what to do if a byte Series is given that
        contains characters not of the given `encoding`. By default,
        it is 'strict', meaning that a UnicodeDecodeError will be
        raised. Other values are 'ignore' and 'replace'.
    pandas.Series:
        A cleaned Series of strings.
    """

    if s.shape[0] == 0:
        return s

    # Lower s if lower is True
    if lowercase is True:
        s = s.str.lower()

    # Accent stripping based on https://github.com/scikit-learn/
    # scikit-learn/blob/412996f/sklearn/feature_extraction/text.py
    # BSD license
    if not strip_accents:
        pass
    elif callable(strip_accents):
        strip_accents_fn = strip_accents
    elif strip_accents == 'ascii':
        strip_accents_fn = strip_accents_ascii
    elif strip_accents == 'unicode':
        strip_accents_fn = strip_accents_unicode
    else:
        raise ValueError(
            "Invalid value for 'strip_accents': {}".format(strip_accents)
        )

    # Remove accents etc
    if strip_accents:
        def strip_accents_fn_wrapper(x):
            if sys.version_info[0] >= 3:
                if isinstance(x, str):
                    return strip_accents_fn(x)
                else:
                    return x
            else:
                if isinstance(x, unicode):  # noqa
                    return strip_accents_fn(x)
                else:
                    return x

        # encoding
        s = s.apply(
            lambda x: x.decode(encoding, decode_error) if
            type(x) == bytes else x)
        s = s.map(lambda x: strip_accents_fn_wrapper(x))

    # Remove all content between brackets
    if remove_brackets is True:
        s = s.str.replace(r'(\[.*?\]|\(.*?\)|\{.*?\})', '', regex=True)

    # Remove special characters
    if replace_by_none:
        s = s.str.replace(replace_by_none, '', regex=True)

    if replace_by_whitespace:
        s = s.str.replace(replace_by_whitespace, ' ', regex=True)

    # Remove multiple whitespaces
    s = s.str.replace(r'\s\s+', ' ', regex=True)

    if remove_connectors:
        s = s.str.replace(r'(\s+)(a|an|and|the|or)(\s+)', ' ', regex=True)

    if replace_whitespace is not None:
        s = s.str.replace(' ', replace_whitespace, regex=False)
    # Strip s
    s = s.str.lstrip().str.rstrip()

    return s


def clean_phone(s):
    """Clean phonenumbers by removing all non-numbers (except +).
    Parameters
    ----------
    s: pandas.Series
        A Series to clean.
    Returns
    -------
    pandas.Series
        A Series with cleaned phonenumbers.
    """

    # Remove all special tokens
    s = s.astype(object).str.replace('[^0-9+]+', '')

    return s


def clean_zip(x):
    pass


def clean_email(x):
    pass

