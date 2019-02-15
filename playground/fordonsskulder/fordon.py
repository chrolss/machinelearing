import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

filepath = 'playground/fordonsskulder/fordonsskulder.csv'
df = pd.read_csv(filepath, encoding='ISO-8859-1', delimiter=';')

# Poor data quality forces us to convert some dtypes

df['Antal gäldenärer'] = df['Antal gäldenärer'].str.replace(' ','')
df['Antal gäldenärer'] = df['Antal gäldenärer'].astype(str).astype(int)
df['Antal mål'] = df['Antal mål'].str.replace(' ','')
df['Antal mål'] = df['Antal mål'].astype(str).astype(int)
df['Belopp'] = df['Belopp'].str.replace(' ','')
df['Belopp'] = df['Belopp'].astype(str).astype(int)

# Create new features
df['medelbelopp'] = df['Belopp'].divide(df['Antal mål'])

# Create a copy of df without "ÖVRIGA" in LÄN
dfc = df[df['Län'] != 'ÖVRIGA']

# Plot some things
_ = plt.xticks(rotation=45)
