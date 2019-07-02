import bokeh.plotting
from bokeh.plotting import output_file, figure, show
import pandas as pd
from sqlalchemy import create_engine

###### SQL ###########

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


##### pandas ########

df['created_at_m'] = df['created_at'].dt.minute



###### bokeh #######

p = figure(x_axis_label='timestamp', y_axis_label='number of tweets')

p.circle(df['created_at'], df['retweet_count'], color='red')

output_file('testplot.html')
show(p)
