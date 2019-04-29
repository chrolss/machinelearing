import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import bokeh.plotting
from bokeh.plotting import output_file, figure, show

# Load the raw data
train_lab_filepath = 'competition/waterpumps/input/training_set_labels.csv'
train_values_filepath = 'competition/waterpumps/input/training_set_values.csv'

labels = pd.read_csv(train_lab_filepath)
df = pd.read_csv(train_values_filepath)

# Clean the data
df = df.drop(columns='recorded_by', axis=0)  # not needed since all entries are identical


# Visual data exploration
vde = pd.merge(labels, df)

# Plot the different management groups for non functional ones
_ = sns.countplot(x='status_group', hue='waterpoint_type', data=vde[vde.status_group == 'non functional'])

# Plot the different management groups for non functional ones
_ = sns.countplot(x='status_group', hue='water_quality', data=vde[vde.status_group == 'non functional'])

# Plot the map of all waterpumps
_ = sns.scatterplot(x='longitude', y='latitude', hue='status_group', data=vde)

# Bokeh plot for visual inspections and storytelling

test = vde.status_group.unique()

# Create the figure: p
p = figure(x_range=test, x_axis_label='status group', y_axis_label='count')

# Add a blue circle glyph to the figure p
p.vbar(x=test, top=[2, 4, 6], width=0.5, color='blue')

# Specify the name of the file
output_file('bokeh_vde_waterpumps.html')

# Display the plot
show(p)

from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure

output_file("bars.html")

# fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
status_groups = vde.status_group.unique()

#years = ['2015', '2016', '2017']
funders = vde.funder.unique()


def create_bokeh_dict(_df, itervar):
    bokeh_dict = {'status_group': _df.status_group.unique()}
    for item in _df[itervar].unique():
        bokeh_dict[item] = [_df[itervar][(_df[itervar] == item) & (_df.status_group == 'functional')].count(),
                            _df[itervar][(_df[itervar] == item) & (_df.status_group == 'non functional')].count(),
                            _df[itervar][(_df[itervar] == item) & (_df.status_group == 'functional needs repair')]
                            .count()]

    return bokeh_dict

plot_dict = create_bokeh_dict(vde, 'funder')

status = vde.status_group.unique()
funder = vde.funder.unique()

x = [ (status, funder) for stat in status for fund in funder ]
count = sum(zip(plot_dict['2015'], plot_dict['2016'], plot_dict['2017']), ()) # like an hstack

#data = {'fruits' : fruits,
#        '2015'   : [2, 1, 4, 3, 2, 4],
#        '2016'   : [5, 3, 3, 2, 4, 6],
#        '2017'   : [3, 2, 4, 4, 5, 3]}

# this creates [ ("Apples", "2015"), ("Apples", "2016"), ("Apples", "2017"), ("Pears", "2015), ... ]
x = [ (fruit, year) for fruit in fruits for year in years ]
counts = sum(zip(data['2015'], data['2016'], data['2017']), ()) # like an hstack

source = ColumnDataSource(data=dict(x=x, counts=counts))

p = figure(x_range=FactorRange(*x), plot_height=250, title="Fruit Counts by Year",
           toolbar_location=None, tools="")

p.vbar(x='x', top='counts', width=0.9, source=source)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None

show(p)










# Visual comparison of functional vs. non-functional for different features
hue_value = 'payment'
_ = plt.subplot(1,2,1)
_ = sns.countplot(x='status_group', hue=hue_value, data=vde[vde.status_group == 'non functional'])  # Plot the different management groups for non functional ones
_ = plt.subplot(1,2,2)
_ = sns.countplot(x='status_group', hue=hue_value, data=vde[vde.status_group == 'functional'])  # Plot the different management groups for non functional ones

# Correlation analysis

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

# Funder data
bigFund = vde['funder'].value_counts() > 10
bigFund = bigFund[bigFund == True]
smallFund = vde['funder'].value_counts() < 11
smallFund = smallFund[smallFund == True]

# Add new feature
bigfunder = vde.funder.apply(lambda x: 1 if x in bigFund else 0)
fdf = pd.DataFrame(bigfunder)
vde = pd.merge(vde, fdf)
vde = pd.concat([vde, fdf], axis=1)

_ = sns.countplot(x='status_group', hue='bigfunder', data=vde)

# Kmeans analysis

model = KMeans(n_clusters=3)

model.fit(df)
labels = model.predict(df)
