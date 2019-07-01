import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
folder_path = 'projects/F1/input/'

# Lazy way of creating the dataframes from .csv-files
# for file in os.listdir(folder_path):
#     name = re.search(r'[A-z]+', file)
#     globals()[name.group(0)] = pd.read_csv(folder_path + file, encoding='latin-1')

lapTimes = pd.read_csv(folder_path + 'laptimes.csv', encoding='latin-1')
races = pd.read_csv(folder_path + 'races.csv', encoding='latin-1')
drivers = pd.read_csv(folder_path + 'drivers.csv', encoding='latin-1')
results = pd.read_csv(folder_path + 'results.csv', encoding='latin-1')
circuits = pd.read_csv(folder_path + 'circuits.csv', encoding='latin-1')

_ = plt.xticks(rotation=45)
_ = sns.countplot(x='nationality', data=drivers)
_ = sns.countplot(x='name', data=races, order=races.name.value_counts())

# Compare two different races in the 2017 season

# RaceId = 974 is Monaco Grand Prix, year 2017
gp = lapTimes[lapTimes.raceId == 974]
_ = plt.subplot(2,1,1)
_ = plt.title('Monaco 2017')
_ = sns.lineplot(x='lap', y='position', data=gp, hue='driverId')

# RaceId = 973 is Barcelona Grand Prix, Year 2017
_ = plt.subplot(2,1,2)
_ = plt.title('Barcelona 2017')
bgp = lapTimes[lapTimes.raceId == 973]
_ = sns.lineplot(x='lap', y='position', data=bgp, hue='driverId')

# RaceId = 976 is Azerbaijan Grand Prix, Year 2017
_ = plt.subplot(2,1,2)
_ = plt.title('Azerbaijan 2017')
bgp = lapTimes[lapTimes.raceId == 976]
_ = sns.lineplot(x='lap', y='position', data=bgp, hue='driverId')


### Setup some investigative KPI
### 1. Number of DNFs
# Looking at "results", we can see that those who DNF have "position = nan"

DNF = 0
temp = results
temp['position'] = results['position'].fillna(0)
for position in temp[temp.raceId == 973].position:
    if position == 0:
        DNF += 1


def count_dnf(_df, _raceid):
    dnf_count = 0
    temp_df = _df
    # temp['position'] = _df['position'].fillna(0)
    for iter_position in temp_df[temp_df.raceId == _raceid].position:
        if iter_position == 0:
            dnf_count += 1

    return dnf_count


### 2. Number of overtakings
# The theory here is that when one driver changes his or her position between two adjacent laps, then an overtaking has
# occurred. Counting the number of occurences this way, and then divide by 2 will give us the number of overtakings
# since 1 overtaking includes one driver advancing one position, while the other loses one.

competing_drivers = []
for driver in lapTimes[lapTimes.raceId == 973].driverId:
    if driver not in competing_drivers:
        competing_drivers.append(driver)

previousPosition = 0
overtakings = 0
for driver in competing_drivers:
    for lapPosition in lapTimes[lapTimes.raceId == 973][lapTimes.driverId == driver].position:
        if lapPosition != previousPosition:
            previousPosition = lapPosition
            overtakings += 1


### 3. Variance in the race positions
# Instead of trying to count the specific number of overtakings, It might be nicer to capture the variance
# in the lap positions in a race. Let's take Barcelona 2017 as an example:


#competing_drivers = []
#for driver in lapTimes[lapTimes.raceId == 973].driverId:
#    if driver not in competing_drivers:
#        competing_drivers.append(driver)
#
#list_of_variance = []
#
#for driver in competing_drivers:
#    temp_var = np.var(lapTimes[lapTimes.raceId == 973][lapTimes.driverId == driver].position)
#    list_of_variance.append(temp_var)


def get_std(_laptimes, _raceId):

    competing_drivers = []
    for driver in _laptimes[_laptimes.raceId == _raceId].driverId:
        if driver not in competing_drivers:
            competing_drivers.append(driver)

    list_of_std = []
    for driver in competing_drivers:
        temp_var = np.std(_laptimes[_laptimes.raceId == _raceId][_laptimes.driverId == driver].position)
        list_of_std.append(temp_var)

    return np.max(list_of_std), np.min(list_of_std), np.mean(list_of_std)

### 4. Top 5 driver battle
# The hypothesis here is that a good battle between the top 5 drivers will result in a higher race rating than one where
# the top 5 never really fight, and the standings remain the same throughout the race.
# One can look at the final score, but it might be better to look and see how the race progressed for those who ended
# the race in position 1 - 5. E.g. Did you start as 1 and finish as 1 ? = BORING. Did you start as 3 but ended as 1?
# = NICE. First idea one might get is to just check the start position and compare to final standing. However, this may
# only show that one person did one pitstop overtaking and won a couple of positions, and then kept that throughout the
# race. A better one is to see "how was the race for those who ended up in 1 - 5 position?" We might actually check the
# variance again, but weight it according to 1 (high weight), 5 (lower weight)

barca_id = 973  # Barcelona
top_5 = results[results.raceId == 973][results.position < 6].sort_values(['position'], ascending=True)
top_5_var = []

for driver in top_5.driverId:
    t5_var = np.var(lapTimes[lapTimes.driverId == driver][lapTimes.raceId == barca_id].position)
    top_5_var.append(t5_var)

top5score = 0
for itervar in top_5_var:
    top5score = top5score + itervar


def get_top_5_battle(_raceid, _results, _laptimes):
    f_top_5 = _results[_results.raceId == _raceid][_results.position < 6].sort_values(['position'], ascending=True)
    f_top_5_var = []

    for f_driver in f_top_5.driverId:
        f_t5_var = np.var(_laptimes[_laptimes.driverId == f_driver][_laptimes.raceId == _raceid].position)
        f_top_5_var.append(f_t5_var)

    f_top5score = 0
    for f_itervar in f_top_5_var:
        f_top5score = f_top5score + f_itervar

    return f_top5score


### 5. Rank vs. position
# Is there a link between "a good race" and the drivers rank?

top5name = pd.merge(top_5, drivers,how='inner',on='driverId')
top5name[['driverId', 'surname','position','rank']]

####### Start the analysis ########
# We will create a dataframe with some original features from other dataframes, as well as some of our own created
# features. The feature engineering part will consist of: number of overtakings, number of DNFs, average speed,
# total race duration, "top 5 fight" (not yet defined, but will tell if one driver is "too superior" to the competition

# We will focus on the 2017 season, so we start by selecting all the races that took place in that season
races17 = races[races.year == 2017]

# We add some "reader-friendly" features to distinguish the races
races17 = pd.merge(races17, circuits, how='left', on='circuitId')

# For easy calculation, we remoove the nan in lapTimes and replace with "0" to represent DNF
# lapTimes['position'] = lapTimes['position'].fillna(0)
results['position'] = results['position'].fillna(0)

### Create the first feature, number of DNF
dnf_list = [count_dnf(results, raceId) for raceId in races17.raceId]
races17['DNF'] = dnf_list


### Create the second features, lap position variances (max, min, mean)

max_std, min_std, mean_std = [], [], []

for raceId in races17.raceId:
    t_max, t_min, t_mean = get_std(lapTimes, raceId)
    max_std.append(t_max)
    min_std.append(t_min)
    mean_std.append(t_mean)

races17['max_std'] = max_std
races17['min_std'] = min_std
races17['mean_std'] = mean_std

# This shows that Azerbaijan has a crazy high max variance, which might be that someone starts in a good position,
# then "breaks" and ends up at a high position?

### Create the third feature - top 5 drivers

top5_score = [get_top_5_battle(raceId, results, lapTimes) for raceId in races17.raceId]
races17['top_5'] = top5_score

# The example race, Spanish Grand Prix 2017 can be found on the wiki site
# https://en.wikipedia.org/wiki/2017_Spanish_Grand_Prix
# From this site, we see that if the winner is more than 1 lap ahead of you when he/she finishes, then you are
# "retired" from the race and you will by default race less laps than them.
# The rating of the races (according to racefans.net), can be found on this page
# https://www.racefans.net/rate-the-race/f1-fanatic-top-100/
#

### The rating for the race was found on "racefans.net", and for the 2017 season I have collected the ratings for each
# race.

# Read race rating values
raceRating = pd.read_csv('projects/F1/input/raceRating.csv', delimiter=';', encoding='latin-1')

# Merge with races17 dataframe
races17 = pd.merge(races17, raceRating, how='inner', on='raceId')
races17 = races17.sort_values(['rating'], ascending=False).reset_index(drop=True)

# Plot the rating races
_ = plt.xticks(rotation=45)
_ = sns.barplot(x='name_x', y='rating', data=races17)