import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

## Load the raw data
train_lab_filepath = 'competition/waterpumps/training_set_labels.csv'
train_values_filepath = 'competition/waterpumps/training_set_values.csv'

labels = pd.read_csv(train_lab_filepath)
df = pd.read_csv(train_values_filepath)

## Clean the data
df = df.drop(columns='recorded_by', axis=0)  # not needed since all entries are identical


## Visual data exploration
vde = pd.merge(labels, df)

_ = sns.countplot(x='status_group', hue='funder', data=vde)  # Plot the different status and management groups
_ = sns.countplot(x='status_group', hue='waterpoint_type', data=vde[vde.status_group == 'non functional'])  # Plot the different management groups for non functional ones
_ = sns.countplot(x='status_group', hue='waterpoint_type', data=vde[vde.status_group == 'non functional'])  # Plot the different management groups for non functional ones
_ = sns.scatterplot(x='longitude', y='latitude', hue='status_group', data=vde)

corr = compmerge.corr()

_ = sns.heatmap(corr)

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
