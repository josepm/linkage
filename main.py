"""
__author__: josep ferrandiz
# https://www.robinlinacre.com/maths_of_fellegi_sunter/

Must define validation and score functions

validate data (
feature eng
run record match

"""
import sys
sys.path.append('/')
import pandas as pd
import sales_match.record_linkage.rl_utils as rlu
import my_utilities.utilities as ut
import sales_match.record_linkage.record_match as rm
import sales_match.record_linkage.clf_utilities as clf


N_JOBS = -1

if __name__ == "__main__":

    # email providers
    # https://gist.github.com/ammarshah/f5c2624d767f91a7cbdc4e54db8dd0bf#file-all_email_provider_domains-txt
    d1 = pd.DataFrame(rlu.dlist, columns=['email_domain'])
    domains = pd.read_csv('~/my_data/matching/all_email_provider_domains.csv')
    d2 = pd.DataFrame(domains).drop_duplicates().dropna().reset_index(drop=True)
    d2.columns = ['email_domain']
    df_ = pd.concat([d1, d2], axis=0)
    df_.drop_duplicates(inplace=True)
    df_['email_domain'] = df_['email_domain'].astype('string[pyarrow]')
    df_.index = df_['email_domain'].values

    # load sales
    sales = pd.read_csv('~/my_data/matching/sales.csv.zip', low_memory=False)
    sales.rename(columns={'email_1': 'email'}, inplace=True)  # email_2 and email_3 always NA
    s_cols = ['first_name', 'last_name', 'city', 'state', 'zip', 'email', 'home_phone', 'mobile_phone', 'work_phone', 'dpid']
    sales = sales[s_cols].copy()
    sales.rename(columns={'home_phone': 'phone1', 'mobile_phone': 'phone2', 'work_phone': 'phone3'}, inplace=True)
    sales.drop_duplicates().reset_index(inplace=True, drop=True)
    sales['sale_id'] = sales.index
    sales['sale_id'] = sales['sale_id'].astype('string[pyarrow]')

    # clean
    # add features

    sales = rlu.data_process(sales.copy(), df_, thres=0.95)  # clean the sales data
    sales.dropna(subset=['first_name', 'last_name'], inplace=True)
    ut.to_parquet(sales, '~/my_data/matching/sales.par')

    # load users
    users = pd.read_csv('~/my_data/matching/agents.csv.zip', low_memory=False)
    u_cols = ['first_name', 'last_name', 'city', 'state', 'zip', 'email', 'phone', 'dpid', 'user_id']
    users = users[u_cols].copy()
    users.drop_duplicates().reset_index(inplace=True, drop=True)
    users['phone1'] = users['phone'].values
    users['phone2'] = users['phone'].values
    users['phone3'] = users['phone'].values
    users.drop('phone', inplace=True, axis=1)

    # clean
    # add features

    users = rlu.data_process(users.copy(), df_, thres=0.95)  # clean the sales data
    users.dropna(subset=['first_name', 'last_name', 'email_name', 'email_domain'], inplace=True)
    ut.to_parquet(users, '~/my_data/matching/users.par')

    users = pd.read_parquet('~/my_data/matching/users.par')
    sales = pd.read_parquet('~/my_data/matching/sales.par')

    # m_cols: dict with scoring info
    # key: [match_op, match_levels, cols]
    # other cols: idx_col, dpid
    m_cols = {
        # 'city': ['string', [0, 0.88, 0.94, 1]],
        # 'state': ['string', [0, 0.88, 0.94, 1]],
        # 'first_name': ['exact', 2],
        # 'last_name': ['exact', 2],
        'cityzip': ['cityzip', 3, ['city', 'zip']],
        'first_name_sound': ['exact', 2, ['first_name_sound']],
        'last_name_sound': ['exact', 2, ['last_name_sound']],
        'email_domain': ['exact', 2, ['email_domain']],
        'email_name': ['string', [0, 0.88, 0.94, 1], ['email_name']],
        # 'zip': ['exact', 2],
        # 'phone1': ['exact', 2],
        # 'phone2': ['exact', 2],
        # 'phone3': ['exact', 2]
    }

    m_cols_ = m_cols
    sf = sales[sales['dpid_clean'] == 'earlstewarttoyota'].copy()
    users_data = users.copy()
    dpid = sf['dpid_clean'].unique()[0]
    uf = users_data[users_data['dpid_clean'] == dpid].copy()

    mm_obj = rm.RecordMatchModel(sf, 'sale_id', uf, 'user_id', m_cols)
    mm_obj.set_index()
    mm_obj.set_scores()
    mm_obj.em()
    mf = mm_obj.record_matches(clf.minFR, 1.0)  # clf.minFbeta, 1.0

    print('match rate: ' + str(len(mf)/len(sf)))


