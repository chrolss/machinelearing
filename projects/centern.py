from learn.twitter.twitter_stream import *
from learn.twitter.twitter_functions import *
import pandas as pd
import re
import matplotlib.pyplot as plt

# Collect the latest tweets

filepath = 'learn/twitter/twitter_keys'
auth = get_auth_token(filepath)
df = get_twitter_search_df(auth, '@centerpartiet', 79, '2019-05-01')

# Start analyzing

hashtag_dict = dict()
pattern = r'[\#][A-z]+'

for i in range(len(df)):
    res = re.findall(pattern, df.text.iloc[i])
    for item in res:
        if item in hashtag_dict:
            hashtag_dict[item] += 1
        else:
            hashtag_dict[item] = 1

lists = sorted(hashtag_dict.items())
x, y = zip(*lists)

_ = plt.bar(x, y)