import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filepath = 'playground/boostmail/emaildomain.csv'

df = pd.read_csv(filepath, index_col=None)

df = df.reset_index()
df = df.drop(columns='emaildomain', axis=0)
df.columns = ['emaildomain']

nunique = df.nunique()
_ = plt.plot(df.emaildomain.value_counts())
