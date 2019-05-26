import nltk as nlp
import tweepy
from tweepy import OAuthHandler
import os

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





# Create a tweepy StreamListener

class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status.text)

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_error disconnects the stream
            return False

        # returning non-False reconnects the stream, with backoff.


myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
myStream.filter(track=['#eu'])

