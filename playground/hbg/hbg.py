import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filepath = 'learn/hbg/input/data.csv'

df = pd.read_csv(filepath)

sns.countplot(df.Händelsetyp, order=df.Händelsetyp.value_counts().iloc[:5].index)

sns.countplot(df[df['Händelsetyp'] == 'Polisrapport']['Rubrik'], order=df[df['Händelsetyp'] == 'Polisrapport']['Rubrik'].value_counts().iloc[:3].index)

sns.countplot(df[df['Händelsetyp'] == 'Polisrapport']['Statistikområde_B'], order=df[df['Händelsetyp'] == 'Polisrapport']['Statistikområde_B'].value_counts().iloc[:7].index)

sns.countplot(df[df['Händelsetyp'] == 'Klotter']['Statistikområde_B'], order=df[df['Händelsetyp'] == 'Klotter']['Statistikområde_B'].value_counts().iloc[:5].index)
