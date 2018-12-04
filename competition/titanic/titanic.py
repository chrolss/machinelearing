import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pt

testFilepath = "competition/titanic/input/test.csv"
trainFilepath = "competition/titanic/input/train.csv"

df = pd.read_csv(trainFilepath)

## working with the data, let's change some

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

## Correlation matrix

sns.set(style='white')
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask,cmap=cmap,vmax=.3,center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

## looking for NANs and nulls, find lots in cabin and age
df.isnull().sum()

## drop cabin and ticket
df = df.drop(['Cabin'], axis=1)
df = df.drop(['Ticket'], axis=1)
## set missing ages as mean age
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna('S')

# One-hot encoding för

## Feature engineering
# 1 - travelling alone
# 2 -


