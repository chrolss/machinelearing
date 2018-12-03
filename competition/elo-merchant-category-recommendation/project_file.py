import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Necessary
merchants = pd.read_csv('competition/elo-merchant-category-recommendation/input/merchants.csv')
new_merchants_transaction = pd.read_csv('competition/elo-merchant-category-recommendation/input/new_merchant_transactions.csv')
historical_transactions = pd.read_csv('competition/elo-merchant-category-recommendation/input/historical_transactions.csv')

# Actual case test data 2
df = pd.read_csv('competition/elo-merchant-category-recommendation/input/train.csv')

## First thing to do seems to be doing some correlation analysis on the merchants to find how they are related

new_merchants_transaction.loc[new_merchants_transaction['merchant_id'] == 'M_ID_b0c793002c']['card_id']
new_merchants_transaction.loc[new_merchants_transaction['merchant_id'] == 'M_ID_b0c793002c'].count()
