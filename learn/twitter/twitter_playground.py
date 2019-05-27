import nltk as nlp
import json
import pandas as pd
from learn.twitter.twitter_functions import get_auth_token
from learn.twitter.twitter_functions import get_tweets_from_user
from learn.twitter.twitter_functions import get_twitter_search_df
from learn.twitter.twitter_stream import deploy_stream_listener_to_df

# Read and save the consumer and access keys

key_file_path = 'learn/twitter/twitter_keys'
api = get_auth_token(key_file_path)

# Test: Read my own timeline and print the latest tweets
public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)


# Play with the stream

deploy_stream_listener_to_df(api, '#svpol')


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

