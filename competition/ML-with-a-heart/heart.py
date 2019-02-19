import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# Create ECDP (Emperical cumulative distribution function)
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

train_values = 'competition/ML-with-a-heart/input/train_values.csv'
train_labels = 'competition/ML-with-a-heart/input/train_labels.csv'

X = pd.read_csv(train_values)
y = pd.read_csv(train_labels)

# Concatenate dataframes to
df = pd.concat([y.heart_disease_present,X],axis=1,join='inner')

# Label Encoding

df.thal = df['thal'].astype('category')
df = pd.concat([df, pd.get_dummies(df.thal,'thal')],axis=1)
df = df.drop(['thal'],axis=1)
# Perform Visual EDA
_ = sns.boxplot(x='heart_disease_present',y='resting_blood_pressure',data=df)

# Perform correlation analysis

sns.set(style='white')
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask,cmap=cmap,vmax=.3,center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Build model

X_train, y_train, X_test, y_test = train_test_split(df.drop(['heart_disease_present'],axis=1),df.heart_disease_present)