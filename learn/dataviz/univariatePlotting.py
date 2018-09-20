import pandas as pd
filepath = "../input/wine_reviews/winemag-data_first150k.csv"
reviews = pd.read_csv(filepath, index_col=0)
