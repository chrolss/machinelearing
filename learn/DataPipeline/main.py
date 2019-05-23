import requests
from sqlalchemy import create_engine
import pandas as pd

# 1. Fetch data from API

url = 'https://brottsplatskartan.se/api/events/?location=stockholms län'
response = requests.get(url)  # Request data from API
body = response.json()        # Save the response in .json format
data = body['data']           # Extract the response data body

# 2. Save data in pandas dataframe

column_names = data[0].keys()
df = pd.DataFrame(data, columns=column_names)

# 3. Create database connection
avo_path = 'mssql+pyodbc://admin:admin@192.168.0.106\\SQLEXPRESS/test?driver=SQL+Server'
test_path = 'mssql+pyodbc://DESKTOP-LUI8MOK\\SQLEXPRESS/test?driver=SQL+Server'

engine = create_engine(avo_path)
con = engine.connect()

# 4. Rename the column labels
new_column_names = ['id', 'pubdate', 'pubdate_unix', 'type', 'city', 'description', 'content_raw', 'content', 'teaser',
                    'location_long', 'date_human', 'lat', 'long', 'vp_ne_lat', 'vp_ne_long', 'vp_sw_lat', 'vp_sw_long', 'area_l1',
                    'area_l2', 'imageurl', 'source_link', 'permalink']
df.columns = new_column_names
# 5a. Use pandas to push into database table (this has the problem that it will write duplicates and ignore
# primary key. A possible solution is to send all rows one by one with a "INSERT OR UPDATE" query

df.to_sql('crime', con=engine, if_exists='append', index=False)
engine.execute("SELECT * FROM crime").fetchall()

# 5b. Use SQL-alchemy to push the new data into the database

for row in range(len(df)):


con.close()


