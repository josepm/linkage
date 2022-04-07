"""
__author__: josep ferrandiz
score functions used to compare left and right data

"""
import pandas as pd
import numpy as np
from jellyfish import jaro_winkler as jw_sim
from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode
from functools import lru_cache


def cmp_exact(s1, s2):
    # s1, s2 pd.Series otherwise elementwise comparisons may fail
    # Note: when dtypes are 'string', pd.NA == <a_string> returns pd.NA
    nulls = np.array(pd.isna(s1) | pd.isna(s2), dtype=pd.UInt8Dtype)  # 1 if s1 or s2 are null
    scores = pd.Series(s1 == s2)
    scores.fillna(False, inplace=True)
    return nulls, scores.values   # delta, gamma


def cmp_string(s1, s2):
    # s1, s2 pd.Series otherwise elementwise comparisons may fail
    # Note: when dtypes are 'string', pd.NA == <a_string> returns pd.NA
    nulls = np.array(pd.isna(s1) | pd.isna(s2), dtype=pd.UInt8Dtype)  # 1 if s1 or s2 are null
    scores = vjw(s1.values, s2.values)
    return nulls, scores


@lru_cache(None)
def jw_func(x1, x2):
    # x1 and x2 are strings
    try:
        return jw_sim(x1, x2)
    except (TypeError, Exception) as e:
        return 0.0   # NAs are captured by delta's


vjw = np.vectorize(jw_func)  # vectorize jw_func


def cmp_cityzip(s1, s2):
    # joint scoring of city and zip
    # s1: DF with cols named 'city' and 'zip'
    # s2: DF with cols named 'city' and 'zip'
    # s1, s2 pd.DF otherwise elementwise comparisons may fail
    # Note: when dtypes are 'string', pd.NA == <a_string> returns pd.NA
    # if zipL == zipR return 1
    # else  # different zips or missing zip(s)
    #       if cityL == cityR return 0.5
    #       else: return 0
    # nulls: isna(cityL) | isna(cityR) because null zips are recovered from city
    nulls = np.array(pd.isna(s1['city_L']) | pd.isna(s2['city_R']), dtype=pd.UInt8Dtype)  # because null zips are recovered from city
    zscores = pd.Series(s1['zip_L'] == s2['zip_R'], dtype='boolean')
    zscores.fillna(False, inplace=True)
    b = zscores == True
    zscores = zscores[b].copy()          # zip matches
    zscores = zscores.astype('float32')  # return 1 on zip matches

    # return lower match value when zip fails but city matches
    cscores = pd.Series(s1[~b]['city_L'] == s2[~b]['city_R'], dtype='boolean')
    cscores.fillna(False, inplace=True)
    cscores = cscores.astype('float32')  # return 1 on zip matches
    cscores *= 0.75                    # scale city matching score to 0.75
    s = pd.concat([zscores, cscores], axis=0)
    return nulls, s.sort_index()


def cmp_phone(s1, s2):
    # https://stackabuse.com/validating-and-formatting-phone-numbers-in-python/
    # https://pypi.org/project/phonenumbers/
    pass


def cmp_email(s1, s2):
    pass


def cmp_num(s1, s2):
    pass


def cmp_dates(s1, s2):
    pass


def cmp_geo(s1, s2):
    pass


