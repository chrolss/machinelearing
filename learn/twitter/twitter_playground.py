import nltk as nlp
import tweepy
from tweepy import OAuthHandler
import json
import pandas as pd

# Read and save the consumer and access keys

key_file_path = 'learn/twitter/twitter_keys'
keys = []
with open(key_file_path) as file:
    keys = file.read().splitlines()

consumer_key = keys[0]
consumer_secret = keys[1]
access_token = keys[2]
access_secret = keys[3]

# Setup OAuthentication and access tokens
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

# Test: Read my own timeline and print the latest tweets
public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)


# Search for tweets with keywords (and maybe hashtags?)

tweets = api.search(q="#sweden", count=10, since="2019-05-20")


def getTwitterSearchToDF(_auth, _searchterm, _nrOfTweets, _earliestDate):
    tweets = api.search(q=_searchterm, count=_nrOfTweets, since=_earliestDate)

    tweetDict = []
    for tweet in tweets:
        tweetDict.append(tweet._json)

    with open('tweet_dump.txt', 'w') as file:
        file.write(json.dumps(tweetDict, indent=4))

    my_demo_list = []
    with open('tweet_dump.txt', encoding='utf-8') as json_file:
        all_data = json.load(json_file)
        for each_dictionary in all_data:
            tweet_id = each_dictionary['id']
            text = each_dictionary['text']
            favorite_count = each_dictionary['favorite_count']
            retweet_count = each_dictionary['retweet_count']
            created_at = each_dictionary['created_at']
            screen_name = each_dictionary['user']['screen_name']
            my_demo_list.append({'tweet_id': str(tweet_id),
                                 'text': str(text),
                                 'favorite_count': int(favorite_count),
                                 'retweet_count': int(retweet_count),
                                 'created_at': created_at,
                                 'user': screen_name
                                 })
        # print(my_demo_list)
        tweet_json = pd.DataFrame(my_demo_list, columns=
        ['tweet_id', 'text',
         'favorite_count', 'retweet_count',
         'created_at', 'user'])

    return tweet_json

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

