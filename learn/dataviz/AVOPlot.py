import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## This file contains some example plots done with AVO
df = pd.read_csv(filePath)

## Show data in easy table form
df.head().T

## Example start
df["loan_condition"].value_counts(normalize=True)
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,5))
colors = ["#0D2C54", "#F6511D"]
g = df["loan_condition"].value_counts(normalize=True).plot(kind='bar', ax=ax[0], fontsize=12, color=colors)
for p in ax[0].patches:
    ax[0].annotate(str(round(p.get_height()*100, 1)) + "%", (p.get_x() + p.get_width()/2, p.get_height() * 1.005), ha='center')

ax[0].set_ylabel('% of Condition of Loans', fontsize=14)

sns.countplot(x="issue_year", hue='loan_condition', data=df, palette=colors)
ax[1].set(ylabel="Count of loans");

## Task 1: Loans per year
sns.countplot(df.issue_year)

## Task 2: Amount
sns.barplot(df.issue_year, df.loan_amnt/df.loan_amnt.count())
sns.barplot(df.issue_year, df.loan_amnt)

## Task 3:

