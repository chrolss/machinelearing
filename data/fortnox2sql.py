from sqlalchemy import create_engine
import pandas as pd

# 1. Fetch data from API

excelfilepath = 'C://BluePrism/fortnox.xlsx'
df = pd.read_excel(excelfilepath)

# 2. Create database connection
fortnox_path = 'mssql+pyodbc://admin:admin@192.168.0.106\\SQLEXPRESS/helsingborg?driver=SQL+Server'
avo_path = 'mssql+pyodbc://admin:admin@192.168.0.106\\SQLEXPRESS/test?driver=SQL+Server'
test_path = 'mssql+pyodbc://DESKTOP-LUI8MOK\\SQLEXPRESS/test?driver=SQL+Server'

engine = create_engine(fortnox_path)
con = engine.connect()

df.columns = new_column_names
# 3. Use pandas to push into database table

df.to_sql('TimeTracking', con=engine, if_exists='replace', index=False)
engine.execute("SELECT * FROM TimeTracking").fetchall()

con.close()


