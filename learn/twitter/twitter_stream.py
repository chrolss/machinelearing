import tweepy
import json


class StreamListenerToDataFrame(tweepy.StreamListener):
    # A stream listener that is supposed to catch statuses and write them to a dataframe

    def __init__(self, _maxtweets):
        super(StreamListenerToDataFrame, self).__init__()
        self.num_tweets = 0
        self.max_tweets = _maxtweets
        self.tweetList = []

    def on_status(self, status):
        if self.num_tweets < self.max_tweets:
            self.num_tweets += 1
            self.tweetList.append(status._json)
            print("Tweet number " + str(self.num_tweets) + " out of " + str(self.max_tweets))
            return True

        else:
            with open('tweet_dump.txt', 'w') as file:
                file.write(json.dumps(self.tweetList, indent=4))

            print("Read all tweets")
            return False

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_error disconnects the stream
            return False

        # returning non-False reconnects the stream, with backoff.


def deploy_stream_listener_to_df(_auth, _track, _maxtweets):
    my_stream_listener = StreamListenerToDataFrame(_maxtweets)
    my_stream = tweepy.Stream(auth=_auth.auth, listener=my_stream_listener)
    my_stream.filter(track=[_track])

    return True
