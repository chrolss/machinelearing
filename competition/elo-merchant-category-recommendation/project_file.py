import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Necessary
merchants = pd.read_csv('competition/elo-merchant-category-recommendation/input/merchants.csv')
new_merchants_transaction = pd.read_csv('competition/elo-merchant-category-recommendation/input/new_merchant_transactions.csv')
historical_transactions = pd.read_csv('competition/elo-merchant-category-recommendation/input/historical_transactions.csv')

# Actual case test data 2
df = pd.read_csv('competition/elo-merchant-category-recommendation/input/train.csv')

## First thing to do seems to be doing some correlation analysis on the merchants to find how they are related
sns.set(style='white')
corr = df.corr()
mask = np.zeros_like(df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask,cmap=cmap,vmax=.3,center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

new_merchants_transaction.loc[new_merchants_transaction['merchant_id'] == 'M_ID_b0c793002c']['card_id']
new_merchants_transaction.loc[new_merchants_transaction['merchant_id'] == 'M_ID_b0c793002c'].count()

## Create new columns in the testdata

df.loc[:,'ActiveMonths'] = pd.Series(1, index=df.index)
df.loc[:,'Today'] = pd.Series('2018-12', index=df.index)
df['ActiveMonths'] = df['Today'] - df['first_active_month']