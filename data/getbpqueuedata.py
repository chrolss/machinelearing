# Import necessary packages
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect

# Create engine and connect
engine = create_engine('mssql+pyodbc://DESKTOP-LUI8MOK\\SQLEXPRESS/Training?driver=SQL+Server')
con = engine.connect()

# Create inspector object and get columns names
inspector = inspect(engine)
headers = inspector.get_columns('BPAWorkQueueItem')
columnnames = []

for row in headers:
    columnnames.append(row['name'])

# Query data and create dataframe from fetchall()
query = 'SELECT * FROM BPAWorkQueueItem'
rs = con.execute(query)
raw = pd.DataFrame(rs.fetchall(), columns=columnnames)

