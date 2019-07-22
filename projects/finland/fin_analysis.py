import pandas as pd


pop_grow = 'projects/finland/input/population_growth.csv'
pg = pd.read_csv(pop_grow, encoding='iso-8859-1', delimiter=';')

df = pd.pivot(pg.columns, pg.Kunta, pg.columns)