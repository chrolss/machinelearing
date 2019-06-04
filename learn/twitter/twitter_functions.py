import pandas as pd
import json
import tweepy
from tweepy import OAuthHandler
import re


def get_auth_token(_filepath):
    # Takes a text file containing four lines of twitter app credential keys and returns
    # a tweepy token to be used in future functions

    key_file_path = _filepath  # Standard filepath 'learn/twitter/twitter_keys'
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

    return api


def get_twitter_search_df(_auth, _searchterm, _nrOfTweets, _earliestDate):
    # Input: _auth = tweepy authentication token, _searchterm = search string, _nrOfTweets = maximum nr of tweets
    # _earliestDate = retrieve tweets from this date and future dates
    # Output: pandas dataframe containing the search result

    tweets = _auth.search(q=_searchterm, count=_nrOfTweets, since=_earliestDate)

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
        df_tweet = pd.DataFrame(my_demo_list, columns=
        ['tweet_id', 'text',
         'favorite_count', 'retweet_count',
         'created_at', 'user'])

    return df_tweet


def get_tweets_from_user(_auth, _user):

    tweets = _auth.user_timeline(_user)

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
            language = each_dictionary['lang']
            my_demo_list.append({'tweet_id': str(tweet_id),
                                 'text': str(text),
                                 'favorite_count': int(favorite_count),
                                 'retweet_count': int(retweet_count),
                                 'created_at': created_at,
                                 'user': screen_name,
                                 'language': language
                                 })
        # print(my_demo_list)
        df_tweet = pd.DataFrame(my_demo_list, columns=
        ['tweet_id', 'text',
         'favorite_count', 'retweet_count',
         'created_at', 'user', 'language'])

    return df_tweet


def tweet_json_to_df(_filepath):
    # I think this is the one that works, since not all tweet objects include the metadata field
    temporary_tweet_list = []
    with open('tweet_dump.txt', encoding='utf-8') as json_file:
        all_data = json.load(json_file)
        for each_dictionary in all_data:
            tweet_id = each_dictionary['id']
            text = each_dictionary['text']
            favorite_count = each_dictionary['favorite_count']
            retweet_count = each_dictionary['retweet_count']
            created_at = each_dictionary['created_at']
            screen_name = each_dictionary['user']['screen_name']
            language = each_dictionary['lang']
            temporary_tweet_list.append({'tweet_id': str(tweet_id),
                                        'text': str(text),
                                         'favorite_count': int(favorite_count),
                                         'retweet_count': int(retweet_count),
                                         'created_at': created_at,
                                         'user': screen_name,
                                         'language': language
                                         })
        # print(my_demo_list)
        df_tweet = pd.DataFrame(temporary_tweet_list, columns=
        ['tweet_id', 'text',
         'favorite_count', 'retweet_count',
         'created_at', 'user', 'language'])

    return df_tweet


def regex_search(_pattern, _series):
    temp_dict = dict()
    for i in range(len(_series)):
        res = re.findall(_pattern, _series.iloc[i])
        for item in res:
            if item in temp_dict:
                temp_dict[item] += 1
            else:
                temp_dict[item] = 1

    return temp_dict