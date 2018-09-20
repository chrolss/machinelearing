import pandas as pd
import matplotlib as plt

filepath = '../input/winemag-data-130k-v2.csv'
df = pd.read_csv(filepath)

df2 = pd.DataFrame({'price' : [1,2,3], 'points' : [4,5,6]})
df2.plot.scatter(x='price', y='points')

