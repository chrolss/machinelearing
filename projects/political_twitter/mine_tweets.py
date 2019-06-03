from sqlalchemy import create_engine
from learn.twitter.twitter_stream_sql import *
from learn.twitter.twitter_functions import *

# engine_path = 'path'
# engine = create_engine(engine_path)
# con = engine.connect()

con = 'https://google.com'
auth = get_auth_token('learn/twitter/twitter_keys')
test = deploy_stream_listener_to_sql(auth, '@realdonaldtrump', 2, con)
