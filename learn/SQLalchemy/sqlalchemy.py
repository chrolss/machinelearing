# Import necessary packages
import pandas as pd
from sqlalchemy import create_engine

# Define engine and connect

engine = create_engine('postgresql+psycopg2://pi:<password>@192.168.0.40:5432/test')
table_names = engine.table_names()
print(table_names)

# Create and setup query
con = engine.connect()
rs = con.execute('SELECT * FROM people')

# Save result in a dataframe

df = pd.DataFrame(rs.fechall())

# Validate that data was received
df.info()

# Close database connection

con.close()