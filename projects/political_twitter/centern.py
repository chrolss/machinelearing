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
pattern1 = r'[\#][A-z]+'
pattern2 = r'[\@][A-z]+'

for i in range(len(df)):
    res = re.findall(pattern2, df.text.iloc[i])
    for item in res:
        if item in hashtag_dict:
            hashtag_dict[item] += 1
        else:
            hashtag_dict[item] = 1


lists = sorted(hashtag_dict.items(), key=lambda tup: tup[1], reverse=True)
x, y = zip(*lists)

nr_of_bars = 5
_ = plt.xticks(rotation=45)
_ = plt.bar(x[:nr_of_bars], y[:nr_of_bars])
