import nltk as nlp
import tweepy
from tweepy import OAuthHandler
import os

key_file_path = 'learn/twitter/twitter_keys'

text = open(key_file_path, 'r')

for line in text:
    print(line)