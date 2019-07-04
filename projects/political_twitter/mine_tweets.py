from sqlalchemy import create_engine, Table, MetaData, ForeignKey, Column
from learn.twitter.twitter_stream_sql import *
from learn.twitter.twitter_functions import *

# Get the SQL credentials
key_file_path = 'projects/political_twitter/sql_credentials'  # Standard filepath 'learn/twitter/twitter_keys'

keys = []
with open(key_file_path) as file:
    keys = file.read().splitlines()

sql_username = keys[0]
sql_password = keys[1]
ip = keys[2]

# Setup sqlengine and define table to write
engine_path = 'mssql+pyodbc://' + sql_username + ':' + sql_password + '@' + ip + '\\SQLEXPRESS/' + 'twitter' + '?driver=SQL+Server'
engine = create_engine(engine_path)
con = engine.connect()
metadata = MetaData()
metadata.reflect(engine)
donald = Table('donaldtrump', metadata)

# Get twitter auth token, and start stream
auth = get_auth_token('learn/twitter/twitter_keys')
test = deploy_stream_listener_to_sql(auth, '@realdonaldtrump', 2, engine, donald)
