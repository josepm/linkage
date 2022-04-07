"""
__author__: josep ferrandiz

"""

import pandas as pd
import recordlinkage as rl
from recordlinkage.preprocessing import clean, phonenumbers, phonetic
from recordlinkage.base import BaseCompareFeature
import time
from joblib import Parallel, delayed
import my_utilities.utilities as ut

N_JOBS = -1


class CompareZipCodes(BaseCompareFeature):
    def _compute_vectorized(self, s1, s2):
        """Compare zipcodes.
        If the zipcodes in both records are identical, the similarity
        is 0. If the first two values agree and the last two don't, then
        the similarity is 0.5. Otherwise, the similarity is 0.
        """
        sim = (s1 == s2).astype(float)  # check if the zipcode are identical (return 1 or 0)
        sim[(sim == 0) & (s1.str[0:2] == s2.str[0:2])] = 0.5  # check the first 2 numbers of the distinct comparisons
        return sim


def data_process(f, ref_df, thres=0.95):
    print('set types to string')
    f.fillna(pd.NA, inplace=True)
    for c in f.columns:
        f[c] = f[c].astype('string[pyarrow]')

    # clean columns
    start = time.time()
    with ut.suppress_stdout_stderr():
        results_list = Parallel(n_jobs=N_JOBS)(delayed(clean)(f[c]) for c in ['first_name', 'last_name', 'city', 'state', 'dpid'])
        for ix, c in enumerate(['first_name', 'last_name', 'city', 'state', 'dpid']):
            f[c + '_clean'] = results_list[ix]
    print('cols cleaned in ' + str(time.time() - start) + 'secs')

    # add sound
    start = time.time()
    with ut.suppress_stdout_stderr():
        results_list = Parallel(n_jobs=N_JOBS)(delayed(phonetic)(f[c + '_clean'], 'soundex', decode_error='ignore') for c in ['first_name', 'last_name', 'city'])
        for ix, c in enumerate(['first_name', 'last_name', 'city']):
            f[c + '_sound'] = results_list[ix]
    print('sound done in ' + str(time.time() - start) + 'ses')

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

    # clean email
    start = time.time()
    with ut.suppress_stdout_stderr():
        f = set_email(f.copy())
        f = email_match(f.copy(), ref_df, thres=thres)
    print('email done in ' + str(time.time() - start) + 'secs')

    print('set final data types')
    for c in f.columns:
        f[c] = f[c].astype('string[pyarrow]')
    f.drop_duplicates().reset_index(inplace=True, drop=True)
    return f


def email_match(mf, ref_df, thres=0.95):
    df = pd.DataFrame(mf['email_domain']).drop_duplicates().dropna().reset_index(drop=True)
    df.index = df['email_domain'].values
    indexer = rl.SortedNeighbourhoodIndex(left_on='email_domain', right_on='email_domain', window=101)
    candidate_links = indexer.index(df, ref_df)
    compare_cl = rl.Compare(n_jobs=-1)
    compare_cl.string('email_domain', 'email_domain', method='jarowinkler', threshold=None, label='email_domain')
    features = compare_cl.compute(candidate_links, df, ref_df)
    features.reset_index(inplace=True)
    features.columns = ['email_domain', 'ref_domain', 'score']
    features.sort_values(by='score', inplace=True, ascending=False)
    features.drop_duplicates(subset=['email_domain'], keep='first', inplace=True)
    matches = features[features['score'] >= thres].copy()
    fout = mf.merge(matches, on='email_domain', how='left')
    fout['email_domain'] = fout['ref_domain'].values
    fout.drop(['ref_domain', 'email', 'score'], axis=1, inplace=True)
    return fout


def nlargest(f, n, columns, keep):
    return f.nlargest(n, columns=columns, keep=keep)


def s_match(sf, users_data, m_cols_):
    dpid = sf['dpid_clean'].unique()[0]
    uf = users_data[users_data['dpid_clean'] == dpid].copy()
    if len(uf) == 0:
        print('no users for ' + str(dpid))
        return pd.DataFrame()
    else:
        cols = list(m_cols_.keys())
        f_users = pd.DataFrame(uf[cols + ['user_id']].drop_duplicates())
        f_users.set_index('user_id', inplace=True)
        f_sales = pd.DataFrame(sf[cols + ['sale_id']].drop_duplicates())
        f_sales.set_index('sale_id', inplace=True)
        indexer = rl.index.Full()
        candidate_links = indexer.index(f_sales, f_users)
        compare_cl = rl.Compare(n_jobs=-1)
        for c in cols:
            if m_cols_[c] == 'exact':
                compare_cl.exact(c, c, label=c)
            elif m_cols_[c] == 'string':
                compare_cl.string(c, c, method='jarowinkler', threshold=None, label=c)
            else:
                print(c + ' ' + str(m_cols_[c]) + ' not implemented')
                m_cols_.pop(c)
        features = compare_cl.compute(candidate_links, f_sales, f_users)
        features.fillna(0, inplace=True)
        for c in cols:
            if features[c].nunique() <= 1:
                features.drop(c, axis=1, inplace=True)
        return features


def a_func(features, sf, uf):
        features = rlu.s_match(sf, users_data, m_cols)
        features['phone'] = features[['phone1', 'phone2', 'phone3']].max(axis=1)
        features.drop(['phone1', 'phone2', 'phone3'], axis=1, inplace=True)
        score_weights = {
             'first_name_sound': 1.0,
             'last_name_sound': 1.0,
             'email_domain': 1.0,
             'email_name': 1.0,
             'zip': 1.0,
             'phone': 1.0
        }
        for c in features.columns:
            m = score_weights.get(c, 1.0)
            features[c] *= m
        s = features.sum(axis=1)
        d = s > s.quantile(0.5)
        features = features[d].copy()
        for c in features.columns:  # restore
            m = score_weights.get(c, 1.0)
            features[c] /= m
        # features = features[(features['last_name_sound'] == 1)].copy()
        clf = rl.ECMClassifier(binarize=0.8, max_iter=1000)
        _ = clf.fit(features)
        probs = pd.DataFrame(clf.prob(features))
        probs.columns = ['p_match']
        for c in features.columns:
            m = score_weights.get(c, 1.0)
            features[c] *= m
        probs['score'] = features.sum(axis=1)
        probs['score'] /= probs['score'].max()  # normalize
        for c in features.columns:
            probs[c + '_score'] = features[c]
        probs.reset_index(inplace=True)
        fprobs = probs.groupby('sale_id').apply(nlargest, n=1, columns=['p_match', 'score'], keep='all').reset_index(drop=True)
        z = fprobs.merge(sf[['first_name', 'last_name', 'zip', 'email_name', 'sale_id']], on='sale_id', how='left')
        zz = z.merge(users[['first_name', 'last_name', 'zip', 'email_name',  'user_id']], on='user_id', how='left', suffixes=('_sales', '_users'))
        zz.sort_values(by='p_match', ascending=False, inplace=True)
        zz.reset_index(inplace=True, drop=True)
        for p in [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]:
            for q in [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]:
                zzp = zz[(zz.p_match > p) & (zz.score > q)].copy()
                # b = (zzp['last_name_sales'] == zzp['last_name_users']) & (zzp['first_name_sales'] == zzp['first_name_users'])
                # bsum = b.sum()
                if len(zzp) <= 1.01 * zzp.sale_id.nunique():
                    print('min p_match: ' + str(p) + ' min score: ' + str(q) +
                          ' % ttl sales: ' + str(np.round(zzp.sale_id.nunique() / sf.sale_id.nunique(), 4))
                          # ' full name matches: ' + str(bsum) +
                          # ' doubles: ' + str(np.round(len(zzp) / zzp.sale_id.nunique(), 2)) +
                          # ' correct matches: ' + str(np.round(bsum / zzp.sale_id.nunique(), 2))
                          )


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


def zip_validate(s):
    min_zip = '00501'
    s.fillna(pd.NA, inplace=True)
    s = s.astype(pd.StringDtype(storage='pyarrow'))
    s = s.str.strip()
    s = s.str[:5]
    db = ((s.str.isdigit()) & (s >= min_zip) & (s.str.len() == 5))
    db.fillna(False, inplace=True)  # pd.NA is digit returns pd.NA
    y_zips = s[db]
    n_zips = s[~db]
    n_zips = pd.Series([pd.NA] * len(n_zips), index=n_zips.index)
    z = pd.concat([y_zips, n_zips], axis=0)
    return z.sort_index()


def date_validate(s):
    s.fillna(pd.NaT, inplace=True)
    return pd.to_datetime(s, errors='coerce')


# official email domains
dlist = [
    'gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'hotmail.co.uk', 'hotmail.fr', 'msn.com', 'yahoo.fr', 'wanadoo.fr', 'orange.fr',
    'comcast.net', 'yahoo.co.uk', 'yahoo.com.br', 'yahoo.co.in', 'live.com', 'rediffmail.com',
    'free.fr', 'gmx.de', 'web.de', 'yandex.ru', 'ymail.com', 'libero.it', 'outlook.com', 'uol.com.br',
    'bol.com.br', 'mail.ru', 'cox.net', 'hotmail.it', 'sbcglobal.net', 'sfr.fr', 'live.fr', 'verizon.net',
    'live.co.uk', 'googlemail.com', 'yahoo.es', 'ig.com.br', 'live.nl', 'bigpond.com', 'terra.com.br', 'yahoo.it',
    'neuf.fr', 'yahoo.de', 'alice.it', 'rocketmail.com', 'att.net', 'laposte.net', 'facebook.com', 'bellsouth.net',
    'yahoo.in', 'hotmail.es', 'charter.net', 'yahoo.ca', 'yahoo.com.au', 'rambler.ru', 'hotmail.de', 'tiscali.it',
    'shaw.ca', 'yahoo.co.jp', 'sky.com', 'earthlink.net', 'optonline.net', 'freenet.de', 't-online.de', 'aliceadsl.fr',
    'virgilio.it', 'home.nl', 'qq.com', 'telenet.be', 'me.com', 'yahoo.com.ar', 'tiscali.co.uk', 'yahoo.com.mx',
    'voila.fr', 'gmx.net', 'mail.com', 'planet.nl', 'tin.it', 'live.it', 'ntlworld.com', 'arcor.de', 'yahoo.co.id',
    'frontiernet.net', 'hetnet.nl', 'live.com.au', 'yahoo.com.sg', 'zonnet.nl', 'club-internet.fr', 'juno.com',
    'optusnet.com.au', 'blueyonder.co.uk', 'bluewin.ch', 'skynet.be', 'sympatico.ca', 'windstream.net', 'mac.com',
    'centurytel.net', 'chello.nl', 'live.ca', 'aim.com', 'bigpond.net.au', '123mail.org', '2-mail.com', '4email.net', '50mail.com',
    '9mail.org', 'aapt.net.au', 'adam.com.au', 'airpost.net', 'allmail.net', 'anonymous.to', 'aol.com', 'asia.com', 'berlin.com',
    'bestmail.us', 'bigpond.com', 'bigpond.com.au', 'bigpond.net.au', 'comcast.net', 'comic.com', 'consultant.com', 'contractor.net',
    'dodo.com.au', 'doglover.com', 'doramail.com', 'dr.com', 'dublin.com', 'dutchmail.com', 'elitemail.org', 'elvisfan.com',
    'email.com', 'emailaccount.com', 'emailcorner.net', 'emailengine.net', 'emailengine.org', 'emailgroups.net',
    'emailplus.org', 'emailsrvr.org', 'emailuser.net', 'eml.cc', 'everymail.net', 'everyone.net', 'excite.com', 'execs.com',
    'exemail.com.au', 'f-m.fm', 'facebook.com', 'fast-email.com', 'fast-mail.org', 'fastem.com', 'fastemail.us', 'fastemailer.com',
    'fastest.cc', 'fastimap.com', 'fastmail.cn', 'fastmail.co.uk', 'fastmail.com.au', 'fastmail.es', 'fastmail.fm', 'fastmail.im',
    'fastmail.in', 'fastmail.jp', 'fastmail.mx', 'fastmail.net', 'fastmail.nl', 'fastmail.se', 'fastmail.to', 'fastmail.tw',
    'fastmail.us', 'fastmailbox.net', 'fastmessaging.com', 'fastservice.com', 'fea.st', 'financier.com', 'fireman.net',
    'flashmail.com', 'fmail.co.uk', 'fmailbox.com', 'fmgirl.com', 'fmguy.com', 'ftml.net', 'galaxyhit.com', 'gmail.com', 'gmx.com',
    'googlemail.com', 'hailmail.net', 'hotmail.co.uk', 'hotmail.com', 'hotmail.fr', 'hotmail.it', 'hushmail.com', 'icloud.com',
    'icqmail.com', 'iinet.net.au', 'imap-mail.com', 'imap.cc', 'imapmail.org', 'inbox.com', 'innocent.com', 'inorbit.com',
    'inoutbox.com', 'internet-e-mail.com', 'internet-mail.org', 'lycos.com', 'me.com', 'mybox.xyz', 'netzero.net',
    'postmaster.co.uk', 'protonmail.com', 'reddif.com', 'runbox.com', 'safe-mail.net', 'sync.xyz', 'thexyz.ca', 'thexyz.co.uk',
    'thexyz.com', 'thexyz.eu', 'thexyz.in', 'thexyz.mobi', 'thexyz.net', 'vfemail.net', 'webmail.wiki', 'xyz.am',
    'yandex.com', 'z9mail.com', 'zilladog.com', 'zooglemail.com', 'amazon.com'
]
