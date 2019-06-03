import tweepy

class StreamListenerToDataFrame(tweepy.StreamListener):
    # A stream listener that is supposed to catch statuses and write them to a dataframe

    def __init__(self, _maxtweets, _engine_connection):
        super(StreamListenerToDataFrame, self).__init__()
        self.num_tweets = 0
        self.max_tweets = _maxtweets
        self.engine_connection = _engine_connection
        self.tweetList = []

    def on_status(self, status):
        if self.num_tweets < self.max_tweets:
            self.num_tweets += 1
            self.tweetList.append(status._json)
            print(status._json['id'])
            # Use this for inspiration when completing this
            # https://stackoverflow.com/questions/31750441/generalised-insert-into-sqlalchemy-using-dictionary/31756880
            print("Tweet number " + str(self.num_tweets) + " out of " + str(self.max_tweets))
            return True

        else:
            #with open('tweet_dump.txt', 'w') as file:
            #    file.write(json.dumps(self.tweetList, indent=4))

            print("Read all tweets")
            return False

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_error disconnects the stream
            return False

        # returning non-False reconnects the stream, with backoff.


def deploy_stream_listener_to_sql(_auth, _track, _maxtweets, _engine_connection):
    my_stream_listener = StreamListenerToDataFrame(_maxtweets, _engine_connection)
    my_stream = tweepy.Stream(auth=_auth.auth, listener=my_stream_listener)
    my_stream.filter(track=[_track])

    return True


