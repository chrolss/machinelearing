from sqlalchemy import create_engine, types
import pandas as pd

# Read the excel file

excel_filepath = '/Users/christian/Dropbox/AAAAAA/Ekonomi/BI/Budget ny.xlsx'

df_raw = pd.read_excel(excel_filepath, sheet_name=1)

new_columns_names = ['category', 'subcategory', 'scenario', 'year', 'month', 'date', 'specification',
                     'amount', 'item', 'creditdebet', 'user']

df_raw.columns = new_columns_names

# Cleaning ?

df = df_raw.dropna(thresh=4)
df.year = df.year.fillna(2019)
df.year = df.year.astype(int)
df.amount = df.amount.astype(float)
# Push to SQLite

engine = create_engine('sqlite:////Users/christian/Dropbox/AAAAAA/Ekonomi/BI/household.db')
con = engine.connect()

query = df.to_sql('economy', con=con, if_exists='replace', index=False, dtype={
    'category': types.NVARCHAR(length=255),
    'subcategory': types.NVARCHAR(length=255),
    'scenario': types.NVARCHAR(length=255),
    'year': types.INTEGER(),
    'month': types.INTEGER(),
    'date': types.NVARCHAR(length=255),
    'specification': types.NVARCHAR(length=255),
    'amount': types.FLOAT(precision=2, asdecimal=True),
    'item': types.NVARCHAR(length=255),
    'creditdebet': types.NVARCHAR(length=255),
    'user': types.NVARCHAR(length=255)
})
