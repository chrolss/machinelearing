import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

## Load the raw data
train_lab_filepath = 'competition/waterpumps/input/training_set_labels.csv'
train_values_filepath = 'competition/waterpumps/input/training_set_values.csv'

labels = pd.read_csv(train_lab_filepath)
df = pd.read_csv(train_values_filepath)

## Clean the data
df = df.drop(columns='recorded_by', axis=0)  # not needed since all entries are identical


## Visual data exploration
vde = pd.merge(labels, df)

_ = sns.countplot(x='status_group', hue='funder', data=vde)  # Plot the different status and management groups
_ = sns.countplot(x='status_group', hue='waterpoint_type', data=vde[vde.status_group == 'non functional'])  # Plot the different management groups for non functional ones
_ = sns.countplot(x='status_group', hue='water_quality', data=vde[vde.status_group == 'non functional'])  # Plot the different management groups for non functional ones
_ = sns.scatterplot(x='longitude', y='latitude', hue='status_group', data=vde)


## Visual comparison of functional vs. non-functional for different features
hue_value = 'payment'
_ = plt.subplot(1,2,1)
_ = sns.countplot(x='status_group', hue=hue_value, data=vde[vde.status_group == 'non functional'])  # Plot the different management groups for non functional ones
_ = plt.subplot(1,2,2)
_ = sns.countplot(x='status_group', hue=hue_value, data=vde[vde.status_group == 'functional'])  # Plot the different management groups for non functional ones

## Correlation analysis

X = pd.concat([vde, pd.get_dummies(vde.status_group, 'status')], axis=1)
X = X.corr()
sns.set(style='white')
mask = np.zeros_like(X, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(X, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Correlation analysis gives that there are some features that are correlate to the pump status,
# with a factor > 0.2 (+/-). Population does not seem to correlate to the pump, and the "higher" up pump
# is geographically located, the more functional it seems to be.
# There appears to be specific regions and districts that are more affected by non-functioning pumps

## Funder data
bigFund = vde['funder'].value_counts() > 10
bigFund = bigFund[bigFund == True]
smallFund = vde['funder'].value_counts() < 11
smallFund = smallFund[smallFund == True]

## Add new feature
bigfunder = vde.funder.apply(lambda x: 1 if x in bigFund else 0)
fdf = pd.DataFrame(bigfunder)
vde = pd.merge(vde, fdf)
vde = pd.concat([vde, fdf], axis=1)

_ = sns.countplot(x='status_group', hue='bigfunder', data=vde)

## Kmeans analysis

model = KMeans(n_clusters=3)

model.fit(df)
labels = model.predict(df)