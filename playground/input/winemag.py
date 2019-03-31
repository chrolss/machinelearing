import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

filepath = 'playground/input/winemag-data-130k-v2.csv'

df = pd.read_csv(filepath)

## Do some nice plots

_ = plt.hist(x='points', data=df, bins=10)
_ = plt.xlabel('Bins')
_ = plt.ylabel('Points')

_ = plt.bar(x='country', data=df.dropna(), height='points')

corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask,cmap=cmap,vmax=.3,center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
