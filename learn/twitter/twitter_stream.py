import tweepy


class StreamListenerToDataFrame(tweepy.StreamListener):
    # A stream listener that is supposed to catch statuses and write them to a dataframe

    def __init__(self, api=None):
        super(StreamListenerToDataFrame, self).__init__()
        self.num_tweets = 0

    def on_status(self, status):
        if self.num_tweets < 5:
            self.num_tweets += 1
            print(status)
            return True
        else:
            return False

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_error disconnects the stream
            return False

        # returning non-False reconnects the stream, with backoff.


def deploy_stream_listener_to_df(_auth, _track):
    my_stream_listener = StreamListenerToDataFrame()
    my_stream = tweepy.Stream(auth=_auth.auth, listener=my_stream_listener)
    my_stream.filter(track=[_track])

    return True
