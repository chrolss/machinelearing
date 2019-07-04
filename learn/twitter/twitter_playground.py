import nltk as nlp
import json
import pandas as pd
import seaborn as sns
from learn.twitter.twitter_functions import get_auth_token
from learn.twitter.twitter_functions import get_tweets_from_user
from learn.twitter.twitter_functions import get_twitter_search_df
from learn.twitter.twitter_stream import deploy_stream_listener_to_df
from learn.twitter.twitter_functions import tweet_json_to_df
import re

# Read and save the consumer and access keys

key_file_path = 'learn/twitter/twitter_keys'
api = get_auth_token(key_file_path)

# Play with the stream

json_dump = 'tweet_dump.txt'
list_of_stream_tweets = deploy_stream_listener_to_df(api, '@realdonaldtrump', 50)
df = tweet_json_to_df(json_dump)


# Some feature engineering

df['is_RT'] = df.text.apply(lambda x: 1 if bool(re.match(r'RT', x)) else 0)

# Plotting

_ = sns.countplot(x='language', data=df)
_ = sns.countplot(x='is_RT', data=df)


# Some RegEx fun stuff

import re
testString = 'RT @John I saw that #sweden has grown lately. Greetings from #suomi!'

# Find user mentions in text
pattern = r'[\@][A-z]+'
res = re.findall(pattern, testString)

# Find hashtags
pattern = r'[\#][A-z]+'
res = re.findall(pattern, testString)


# Create a dict of hashtags to use for plotting
hashtag_dict = dict()
pattern = r'[\#][A-z]+'

for i in range(len(df)):
    res = re.findall(pattern, df.text.iloc[i])
    for item in res:
        if item in hashtag_dict:
            hashtag_dict[item] += 1
        else:
            hashtag_dict[item] = 1


# Test: Read my own timeline and print the latest tweets
public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)