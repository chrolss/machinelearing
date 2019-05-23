from sqlalchemy import engine
import pandas as pd
from sqlalchemy import create_engine

## Open and import Excel file data

excelfilepath = 'C://BluePrism/budget.xlsx'
df = pd.read_excel(excelfilepath)

## Clean and edit the data



## Push dataframe to SQL database

mssql_path = 'mssql+pyodbc://DESKTOP-LUI8MOK\\SQLEXPRESS/Finance?driver=SQL+Server'
avo_path = 'mssql+pyodbc://admin:admin@192.168.0.106\\SQLEXPRESS/test?driver=SQL+Server'
sqlite_path = ''

engine = create_engine(avo_path)
con = engine.connect()

testQuery = "INSERT INTO expenses VALUES ('2019-04-13', 'ICA Liljeholmen', '123456', 250.00, 'Food')"
rs = con.execute(testQuery)

## Close and clean-up

con.close()

