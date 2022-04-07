"""
__author__: josep ferrandiz
data validation functions for left and right data sets
"""

import pandas as pd

# official email domains
e_domains = [
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

state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ',
    'Arkansas': 'AK', 'California': 'CA', 'Colorado': 'CO',
    'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL',
    'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY',
    'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MS', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NE', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'NC', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'VA', 'Washington_DC': 'DC',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

area_codes = {'Alabama': [205, 251, 256, 334, 938],
              'Alaska': [907],
              'Arizona': [480, 520, 602, 623, 928],
              'Arkansas': [479, 501, 870],
              'California': [209, 213, 279, 310, 323, 408, 415, 424, 442, 510, 530, 559, 562, 619, 626, 628, 650, 657, 661, 669, 707, 714, 747, 760, 805, 818, 820, 831, 858, 909, 916, 925, 949, 951],
              'Colorado': [303, 719, 720, 970],
              'Connecticut': [203, 475, 860, 959],
              'Delaware': [302],
              'Florida': [239, 305, 321, 352, 386, 407, 561, 727, 754, 772, 786, 813, 850, 863, 904, 941, 954],
              'Georgia': [229, 404, 470, 478, 678, 706, 762, 770, 912],
              'Hawaii': [808],
              'Idaho': [208, 986],
              'Illinois': [217, 224, 309, 312, 331, 618, 630, 708, 773, 779, 815, 847, 872],
              'Indiana': [219, 260, 317, 463, 574, 765, 812, 930],
              'Iowa': [319, 515, 563, 641, 712],
              'Kansas': [316, 620, 785, 913],
              'Kentucky': [270, 364, 502, 606, 859],
              'Louisiana': [225, 318, 337, 504, 985],
              'Maine': [207],
              'Maryland': [240, 301, 410, 443, 667],
              'Massachusetts': [339, 351, 413, 508, 617, 774, 781, 857, 978],
              'Michigan': [231, 248, 269, 313, 517, 586, 616, 734, 810, 906, 947, 989],
              'Minnesota': [218, 320, 507, 612, 651, 763, 952],
              'Mississippi': [228, 601, 662, 769],
              'Missouri': [314, 417, 573, 636, 660, 816],
              'Montana': [406],
              'Nebraska': [308, 402, 531],
              'Nevada': [702, 725, 775],
              'New Hampshire': [603],
              'New Jersey': [201, 551, 609, 640, 732, 848, 856, 862, 908, 973],
              'New Mexico': [505, 575],
              'New York': [212, 315, 332, 347, 516, 518, 585, 607, 631, 646, 680, 716, 718, 838, 845, 914, 917, 929, 934],
              'North Carolina': [252, 336, 704, 743, 828, 910, 919, 980, 984],
              'North Dakota': [701],
              'Ohio': [216, 220, 234, 330, 380, 419, 440, 513, 567, 614, 740, 937],
              'Oklahoma':	[405, 539, 580, 918],
              'Oregon': [458, 503, 541, 971],
              'Pennsylvania': [215, 223, 267, 272, 412, 445, 484, 570, 610, 717, 724, 814, 878],
              'Rhode Island': [401],
              'South Carolina': [803, 843, 854, 864],
              'South Dakota': [605],
              'Tennessee': [423, 615, 629, 731, 865, 901, 931],
              'Texas': [210, 214, 254, 281, 325, 346, 361, 409, 430, 432, 469, 512, 682, 713, 726, 737, 806, 817, 830, 832, 903, 915, 936, 940, 956, 972, 979],
              'Utah': [385, 435, 801],
              'Vermont': [802],
              'Virginia': [276, 434, 540, 571, 703, 757, 804],
              'Washington': [206, 253, 360, 425, 509, 564],
              'Washington_DC': [202],
              'West Virginia': [304, 681],
              'Wisconsin': [262, 414, 534, 608, 715, 920],
              'Wyoming': [307]
              }


def address_validate():
    pass


def phone_validate(s):
    # US only
    s.fillna(pd.NA, inplace=True)
    s = s.astype(pd.StringDtype())
    s = s.str.replace(r'[^0-9]+', '', regex=True)

    # assume 10 digits including area code: drop country code if any
    s = s.str[-10:]
    b = s.str.len() == 10
    b.fillna(False, inplace=True)
    sb = s[b].copy()
    sn = pd.Series([pd.NA] * len(s[~b]), index=s[~b].index)
    zz = pd.concat([sb, sn], axis=0)
    return zz.sort_index()  # put back in index order


def string_validate():
    pass


def zip_validate(s):
    # while we do not get a list of valid state, city, zips
    min_zip = '00501'
    s = s.str.split('-', expand=True)[0]
    b = (s < min_zip) | (s.isnull()) | (s.str.len() < 5)      # bad zips
    b.fillna(False, inplace=True)
    ns = pd.Series([pd.NA] * len(s[b]), index=s[b].index)
    ss = s[~b]
    d = ss.str.isdigit()
    d.fillna(False, inplace=True)
    nd = pd.Series([pd.NA] * len(ss[~d]), index=ss[~d].index)  # bad digits
    zz = pd.concat([ss[d], nd, ns], axis=0)
    return zz.sort_index()  # put back in index order


def date_validate(s):
    s.fillna(pd.NaT, inplace=True)
    return pd.to_datetime(s, errors='coerce')


def edomain_validate():
    pass
