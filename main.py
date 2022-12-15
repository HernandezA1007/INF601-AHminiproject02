# INF601 - Advanced Programming in Python
# Antonio Hernandez
# Mini Project 2


# Proper import of packages used.
import pandas as pd
import matplotlib.pyplot as plt
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import os
import seaborn as sns  # part of matplotlib?

# Using data source: https://www.kaggle.com/datasets/brenda89/fifa-world-cup-2022, retrieve some data for
# creating basic statistics on.
# Kaggle API token...
'''
api = KaggleApi()
api.authenticate()
api.dataset_download_files('brenda89/fifa-world-cup-2022', file_name='international_matches.csv')
'''
# https://www.kaggle.com/questions-and-answers/250858
# https://youtu.be/fmvbY3zkVXc
# download file...
df = pd.read_csv('/kaggle/input/fifa-world-cup-2022/international_matches.csv')
# Store this information in Pandas dataframe. These should be 2D data as a dataframe, meaning the data is labeled
# tabular data.

fifa_team_rank = df[['date', 'home_team', 'away_team', 'home_team_fifa_rank', 'away_team_fifa_rank',
                     'away_team_total_fifa_points', 'home_team_total_fifa_points']]
home = fifa_team_rank[['date', 'home_team', 'home_team_fifa_rank']].rename(columns={'home_team': 'team',
                                                                                    'home_team_fifa_rank': 'rank'})
away = fifa_team_rank[['date', 'away_team', 'away_team_fifa_rank']].rename(columns={'away_team': 'team',
                                                                                    'away_team_fifa_rank': 'rank'})
fifa_team_rank = home.append(away)

def home_percentage(team):
    score = len(df[(df['home_team'] == team) & (df['home_team_result'] == "Win")]) / len(df[df['home_team'] == team]) * 100
    return round(score)
def away_percentage(team):
    score = len(df[(df['away_team'] == team) & (df['home_team_result'] == "Lose")]) / len(df[df['away_team'] == team]) * 100
    return round(score)
# Who are the top 5 teams?
fifa_team_rank = fifa_team_rank.sort_values(['team', 'date'], ascending=[True, False])
fifa_team_rank['row_number'] = fifa_team_rank.groupby('team').cumcount() + 1
top_fifa_teams = fifa_team_rank[fifa_team_rank['row_number'] == 1].drop('row_number').nsmallest(5, 'rank')

print(top_fifa_teams)

# Do home teams win more/have advantage?
# (with how much home win percentage?)
top_fifa_teams['Home_win_Per'] = np.vectorize(home_percentage)(fifa_rank_top10['team'])
top_fifa_teams['Away_win_Per'] = np.vectorize(away_percentage)(fifa_rank_top10['team'])

# Who has the longest win streak?
dframes = {}
for team in fifa_team_rank:
    info = df[(df['home_team'] == team) | (df['away_team'] == team)]
    info['winns'] = np.where(((info['home_team'] == team) & (info['home_team_result'] == 'Win')) | (
            (info['away_team'] == team) & (info['home_team_result'] == 'Lose')), 1, 0)

    name = team + '_streak'
    dframes[name] = info

# Information on 2022 World cup teams:
wc_rank = world_2022[['date','home_team','away_team','home_team_fifa_rank', 'away_team_fifa_rank']]
wc_home = wc_rank[['date','home_team','home_team_fifa_rank']].rename(columns={"home_team":
                                                                            "team","home_team_fifa_rank":"rank"})
wc_away = wc_rank[['date','away_team','away_team_fifa_rank']].rename(columns={"away_team":
                                                                            "team","away_team_fifa_rank":"rank"})

# World Cup... # example from kaggle
fifa2022_teams = ['Qatar', 'Ecuador', 'Senegal', 'Netherlands',
                  'England', 'IR Iran', 'USA', 'Wales',
                  'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
                  'France', 'Australia', 'Denmark', 'Tunisia',
                  'Spain', 'Costa Rica', 'Germany', 'Japan',
                  'Belgium', 'Canada', 'Morocco', 'Croatia',
                  'Brazil', 'Serbia', 'Switzerland', 'Cameroon',
                  'Portugal', 'Ghana', 'Uruguay', 'Korea Republic']

world_2022 = df[(df["home_team"].apply(lambda x: x in fifa2022_teams)) | (df["away_team"]
                                                                          .apply(lambda x: x in fifa2022_teams))]

# Using matplotlib, graph this data in a way that will visually represent the data. Really try to build some fancy
# charts here as it will greatly help you in future homework assignments and in the final project.
streaks = []
for frame in dframes.keys():
    dframes[frame]['start_of_streak'] = dframes[frame]['winns'].ne(dframes[frame]['winns'].shift())
    dframes[frame]['streaks_id'] = dframes[frame]['start_of_streak'].cumsum()
    dframes[frame]['streak_counter'] = dframes[frame].groupby('streaks_id').cumcount() + 1
    streak = dframes[frame][dframes[frame]['winns'] == 1]['streak_counter'].max()
    streaks.append(streak)

Streaks = pd.DataFrame({'Team': fifa_team_rank, 'Streak': streaks}).sort_values('Streak',
                                                                                ascending=False).reset_index(drop=True)
Streaks.index += 1
print(Streaks)

plt.figure(figsize=(11, 7), dpi=90)
ax = sns.barplot(data=Streaks[:10], x='Team', y='Streak')
ax.bar_label(ax.containers[0])
plt.xlabel('Team')
plt.ylabel('Streak')
plt.title('10 longest streaks are: ')

# https://www.kaggle.com/code/prashant111/matplotlib-tutorial-for-beginners/notebook

# Save these graphs in a folder called charts as PNG files. Do not upload these to your project folder,
# the project should save these when it executes. You may want to add this folder to your .gitignore file.

# place questions in a function
# print questions then gives image
# main.py unformatted and testing/experimenting

# create_chart taken from project 1, remake for every question representation
def create_chart(df):
    # check whether charts folder is created, if not make directory

    # creates a numpy array
    for ticker in tickers:
        myData = getStockData(ticker)
        myPrices = np.array(stockData(myData))
        # create matplotlib graph (title, x and y labels)
        plt.plot(myPrices)
        plt.title(ticker)
        plt.xlabel("Day")
        plt.ylabel("Price $")
        # save the graph
        plt.savefig('charts/' + ticker + '.png', facecolor='auto', edgecolor='auto', pad_inches=0.25)
        # show the graph
        plt.show()


# make charts directory if not already made
if not os.path.exists('charts/'):
    os.makedirs('charts/')
