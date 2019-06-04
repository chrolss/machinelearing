from learn.twitter.twitter_stream import *
from learn.twitter.twitter_functions import *
import pandas as pd
import re
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Load tweets from SQL

# Get the SQL credentials
key_file_path = 'projects/political_twitter/sql_credentials'  # Standard filepath 'learn/twitter/twitter_keys'
keys = []
with open(key_file_path) as file:
    keys = file.read().splitlines()

sql_username = keys[0]
sql_password = keys[1]
ip = keys[2]

# Setup sqlengine and define table to write
engine_path = 'mssql+pyodbc://' + sql_username + ':' + sql_password + '@' + ip + '\\SQLEXPRESS/' + 'twitter' \
              + '?driver=SQL+Server'
engine = create_engine(engine_path)
con = engine.connect()

df = pd.read_sql_query('SELECT * FROM donaldtrump', engine)

# Start analyzing

pattern1 = r'[\#][A-z]+'
pattern2 = r'[\@][A-z]+'
hashtag_dict = regex_search(pattern1, df.text)
mention_dict = regex_search(pattern2, df.text)

lists = sorted(mention_dict.items(), key=lambda tup: tup[1], reverse=True)
x, y = zip(*lists)

nr_of_bars = 10
_ = plt.xticks(rotation=45)
_ = plt.bar(x[:nr_of_bars], y[:nr_of_bars])

_ = plt.hist(x='created_at', data=df, bins=100)