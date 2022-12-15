# INF601 - Advanced Programming in Python
# Antonio Hernandez
# Mini Project 2


# Proper import of packages used.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns  # part of matplotlib

# make charts directory if not already made
if not os.path.exists('charts/'):
    os.makedirs('charts/')

# Using data source: https://www.kaggle.com/datasets/brenda89/fifa-world-cup-2022, retrieve some data for
# creating basic statistics on.
# download file...
df = pd.read_csv('international_matches.csv')  # won't load/open no matter what path..?

# Store this information in Pandas dataframe. These should be 2D data as a dataframe, meaning the data is labeled
# tabular data.
fifa_team_rank = df[['date', 'home_team', 'away_team', 'home_team_fifa_rank', 'away_team_fifa_rank',
                     'away_team_total_fifa_points', 'home_team_total_fifa_points']]
home = fifa_team_rank[['date', 'home_team', 'home_team_fifa_rank']] \
                        .rename(columns={'home_team': 'team', 'home_team_fifa_rank': 'rank'})
away = fifa_team_rank[['date', 'away_team', 'away_team_fifa_rank']] \
                        .rename(columns={'away_team': 'team', 'away_team_fifa_rank': 'rank'})

# Who are the top 5 teams? top 10? top 20?
fifa_team_rank = pd.concat([home, away])
top_fifa_teams = fifa_team_rank.groupby('team').first().sort_values('rank', ascending=True)[0:5].reset_index()

# matplotlib
plt.figure(figsize=(10, 4), dpi=200)
plt.title("Top 5 teams")
# plt.xlabel("Rank")
# plt.ylabel("Team")
sns.barplot(data=top_fifa_teams, y='Team', x='Rank')
# save the graph
plt.savefig('charts/' + fifa_team_rank + '.png')
# show the graph
plt.show()


# Who has the longest win streak? (Any country in the list)
# Percentage
def home_percentage(team):
    score = len(df[(df['home_team'] == team) & (df['home_team_result'] == "Win")]) / len(
        df[df['home_team'] == team]) * 100
    return round(score)


def away_percentage(team):
    score = len(df[(df['away_team'] == team) & (df['home_team_result'] == "Lose")]) / len(
        df[df['away_team'] == team]) * 100
    return round(score)


fifa_team_rank['Home_win_Per'] = np.vectorize(home_percentage)(fifa_team_rank['team'])
fifa_team_rank['Away_win_Per'] = np.vectorize(away_percentage)(fifa_team_rank['team'])
fifa_team_rank['Average_win_Per'] = round((fifa_team_rank['Home_win_Per'] + fifa_team_rank['Away_win_Per']) / 2)
fifa_rank_win = fifa_team_rank.sort_values('Average_win_Per', ascending=False)

# matplotlib
plt.title("Top 10 teams w/ Winning Percentage Average")
sns.barplot(data=top_fifa_teams, y='Team', x='Average Winning Percentage')
# save the graph
plt.savefig('charts/' + fifa_team_rank + '.png')
# show the graph
plt.show()

# Top "score" based on FIFA game


# Information on 2022 World cup teams:

# World Cup...
fifa2022_teams = ['Qatar', 'Ecuador', 'Senegal', 'Netherlands',  # Group A
                  'England', 'IR Iran', 'USA', 'Wales',  # Group B
                  'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',  # Group C
                  'France', 'Australia', 'Denmark', 'Tunisia',  # Group D
                  'Spain', 'Costa Rica', 'Germany', 'Japan',  # Group E
                  'Belgium', 'Canada', 'Morocco', 'Croatia',  # Group F
                  'Brazil', 'Serbia', 'Switzerland', 'Cameroon',  # Group G
                  'Portugal', 'Ghana', 'Uruguay', 'Korea Republic']  # Group H

world_2022 = df[(df["home_team"].apply(lambda x: x in fifa2022_teams)) | (df["away_team"]
                                                                          .apply(lambda x: x in fifa2022_teams))]

# Best teams in World Cup 2022 based on winning percentage
wc_rank = world_2022[['date','home_team','away_team','home_team_fifa_rank', 'away_team_fifa_rank']]
wc_home = wc_rank[['date','home_team','home_team_fifa_rank']].rename(columns={"home_team":
                                                                            "team","home_team_fifa_rank":"rank"})
wc_away = wc_rank[['date','away_team','away_team_fifa_rank']].rename(columns={"away_team":
                                                                            "team","away_team_fifa_rank":"rank"})
wcrank = wc_rank.sort_values(['team','date'],ascending=[True,False])
best_wc = wc_rank.groupby('team').first().sort_values('rank',ascending=True).reset_index()
best_wc = best_wc[(best_wc["team"].apply(lambda x: x in fifa2022_teams))][0:10]

best_wc['Home_win_Per'] = np.vectorize(home_percentage)(best_wc['team'])
best_wc['Away_win_Per'] = np.vectorize(away_percentage)(best_wc['team'])
best_wc['Average_win_Per'] = round((best_wc['Home_win_Per'] + best_wc['Away_win_Per'])/2)
best_wc_rank = best_wc.sort_values('Average_win_Per',ascending=False)

# matplotlib
plt.title("Best World Cup Teams based on Winning %")
sns.barplot(data=best_wc_rank, y='Team', x='Average Winning Percentage')
plt.xticks(rotation=90)
# save the graph
plt.savefig('charts/' + best_wc_rank + '.png')
# show the graph
plt.show()

# Using matplotlib, graph this data in a way that will visually represent the data. Really try to build some fancy
# charts here as it will greatly help you in future homework assignments and in the final project.

# Save these graphs in a folder called charts as PNG files. Do not upload these to your project folder,
# the project should save these when it executes. You may want to add this folder to your .gitignore file.

# ... move all plt charts here ...

print("The images have been made and placed in the charts folder")
