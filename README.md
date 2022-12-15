 - INF601 - Advanced Programming in Python
 - Antonio Hernandez
 - Mini Project 2

# INF601 Mini Project 2


### This mini project allows the user to receive images and information from a data set.
The data used can be found https://www.kaggle.com/datasets/brenda89/fifa-world-cup-2022. The type of data in here is
about the FIFA World Cup which is the most prestigious and known tournament in the world. Information included is all 
match history played since the 90s. As well as rankings of each team such as strength of teams based on offense,
defense, and midfield players, home and away team difference, and more. 

---
## Quick Start

> Before starting, look at requirements.txt to install the necessary packages

`pip install -r requirements.txt`

Following questions asked are:  
- `Who are the top 5 teams?`
- `Do home teams win more/have advantage?`
- `Who has the longest win streak?`
- `Information on 2022 World cup teams: `
- `World Cup question...`

> You can change the teams used in fifa2022_teams or
> make your own. For more information on teams...<>


To start up the project, you can:
  - Copy the main.py 
  - Make your own
  - Check out other examples in kaggle...

Example of how some code looks below: 
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

df = pd.read_csv('/kaggle/input/fifa-world-cup-2022/international_matches.csv')

# Who are the top 5 teams?
fifa_team_rank = df[['date', 'home_team', 'away_team', 'home_team_fifa_rank', 'away_team_fifa_rank',
                     'away_team_total_fifa_points', 'home_team_total_fifa_points']]
# code...
top_fifa_teams = fifa_team_rank.groupby('team').first().sort_values('rank', ascending=True)[0:5].reset_index()

print(top_fifa_teams)
# create chart
plt.title("Top 10 teams w/ Winning Percentage Average")
sns.barplot(data=top_fifa_teams, y='Team', x='Average Winning Percentage')
# save the graph
plt.savefig('charts/' + fifa_team_rank + '.png')

...


```

If using Kaggle API...
1. Create/Login account
2. Get/Create your API token in account
3. Download and setup...


## Example of result/output
image...