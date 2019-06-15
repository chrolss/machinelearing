import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
folder_path = 'projects/F1/input/'

for file in os.listdir(folder_path):
    name = re.search(r'[A-z]+', file)
    globals()[name.group(0)] = pd.read_csv(folder_path + file, encoding='latin-1')

_ = plt.xticks(rotation=45)
_ = sns.countplot(x='nationality', data=drivers)
_ = sns.countplot(x='name', data=races, order=races.name.value_counts())

# Compare two different races in the 2017 season

# RaceId = 974 is Monaco Grand Prix, year 2017
gp = lapTimes[lapTimes.raceId == 974]
_ = plt.subplot(2,1,1)
_ = plt.title('Monaco 2017')
_ = sns.lineplot(x='lap', y='position', data=gp, hue='driverId')

# RaceId = 933 is Barcelona Grand Prix, Year 2017
_ = plt.subplot(2,1,2)
_ = plt.title('Barcelona 2017')
bgp = lapTimes[lapTimes.raceId == 973]
_ = sns.lineplot(x='lap', y='position', data=bgp, hue='driverId')


### Setup some investigative KPI
# 1. Number of DNFs
# Looking at "results", we can see that those who DNF have "position = nan"

DNF = 0
temp = results
temp['position'] = results['position'].fillna(0)
for position in temp[temp.raceId == 973].position:
    if position == 0:
        DNF += 1

def count_dnf(_df, _raceId):
    DNF = 0
    temp = _df
    # temp['position'] = _df['position'].fillna(0)
    for position in temp[temp.raceId == _raceId].position:
        if position == 0:
            DNF += 1

    return DNF


# 2. Number of overtakings
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

# Divide the overtakings varibable with 2 and you have the number of totala overtakings. However, since each driver
# improves their "lap-position" if another driver quits the race, then these "improvements" should not be included.
# To account for this, one needs to count "how many drivers improved their lap-position due to a DNF ?

# Start the analysis
# We will create a dataframe with some original features from other dataframes, as well as some of our own created
# features. The feature engineering part will consist of: number of overtakings, number of DNFs, average speed,
# total race duration, "top 5 fight" (not yet defined, but will tell if one driver is "too superior" to the competition

# We will focus on the 2017 season, so we start by selecting all the races that took place in that season
races17 = races[races.year == 2017]

# We add some "reader-friendly" features to distinguish the races
races17 = pd.merge(races17, circuits ,how='left',on='circuitId')

# For easy calculation, we remoove the nan in lapTimes and replace with "0" to represent DNF
lapTimes['position'] = lapTimes['position'].fillna(0)

# Create the first feature, number of DNF
dnf_list = [count_dnf(results, raceId) for raceId in races17.raceId]
races17['DNF'] =
