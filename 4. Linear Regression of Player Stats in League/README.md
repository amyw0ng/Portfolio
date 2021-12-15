# Introduction
## Exploring the Worlds 2021 Play-In Tournament Data

For this project, we're exploring the Group Stage data from the Worlds 2021 League of Legends Play-In matches. We want to identify the player stat category that seems to correlate most with a team's success in the tournament and use a linear regression model to identify other factors that contribute to higher stats in that particular category.

**Hypothesis**: Our hypothesis is that team objectives play the biggest part in a team's success, which we will measure based on the team objective's contribution to the player stat that shows the most distinct difference between players that won or loss from our exploratory data analysis.

<img width="600" height="" align="left" src="https://www.touchtapplay.com/wp-content/uploads/2021/07/league-of-legends-map.jpg">

```python
# First, let's import all the necessary tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_1samp, ttest_ind
```

## Dataset

This dataset was taken from https://www.kaggle.com/braydenrogowski/league-of-legends-worlds-2021-playin-group-stats

Additional information was also taken from https://lol.fandom.com/wiki/2021_Season_World_Championship/Play-In/Scoreboards 

The dataset was combined in Excel and shows the player's individual game stats per match such as kills, deaths, assists and gold earned. In addition, data is also provided on the team's stats for objectives such as Dragons, Barons and Heralds. Please note that the team objective data is duplicated for each player on the same team for the same match.

### Column Legend:
- **Team/Team Name** - The player's team
- **Player** - Player's game ID
- **Opponent/Opponent Name** - The opposing team
- **Position** - The player's role in the game
- **Champion** - The game character that the player was using for that match
- **Kills** - How many times a player landed the killing blow to an opposing player in that match
- **Deaths** - How many times a player died in that match
- **Assists** - How many times a player assisted in killing an opposing player in that match
- **Creep Score** - How many minions/jungle monsters a player landed the killing blow for in that match
- **Gold Earned** - How much gold did that player earn in that match
- **Champion Damage Share** - Percentage share of damage dealt to opposing champions by player in that match
- **Kill Participation** - Percentage of participation in overall Kills or Assists by player in that match
- **Wards Placed** - Total number of wards placed on the map
- **Wards Destroyed** - Total number of opposing team's wards destroyed
- **Ward Interactions** - Total number of times player interacted with enemy wards
- **Dragons For** - Total number of Dragons player's team acquired in that match
- **Dragons Against** - Total number of Dragons opposing team acquired in that match
- **Barons For** - Total number of Barons player's team acquired in that match
- **Barons Against** - Total number of Barons opposing team acquired in that match
- **Result** - Whether the player's team won ("W") or loss ("L") the match
- **Herald For** - Total number of Heralds player's team acquired in that match
- **Herald Against** - Total number of Heralds opposing team acquired in that match

```python
# Let's load in our data
WorldsData = pd.read_excel('League_PlayIn_Data.xlsx', 'PlayInGroupsData')
WorldsData.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Team Name</th>
      <th>Player</th>
      <th>Opponent</th>
      <th>Opponent Name</th>
      <th>Position</th>
      <th>Champion</th>
      <th>Kills</th>
      <th>Deaths</th>
      <th>Assists</th>
      <th>...</th>
      <th>Wards Placed</th>
      <th>Wards Destroyed</th>
      <th>Ward Interactions</th>
      <th>Dragons For</th>
      <th>Dragons Against</th>
      <th>Barons For</th>
      <th>Barons Against</th>
      <th>Result</th>
      <th>Herald For</th>
      <th>Herald Against</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>UOL</td>
      <td>Unicorns of Love</td>
      <td>Boss</td>
      <td>GS</td>
      <td>Galatasaray Esports</td>
      <td>Top</td>
      <td>Camille</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>8</td>
      <td>8</td>
      <td>16</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>L</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GS</td>
      <td>Galatasaray Esports</td>
      <td>Crazy</td>
      <td>UOL</td>
      <td>Unicorns of Love</td>
      <td>Top</td>
      <td>Gwen</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>...</td>
      <td>10</td>
      <td>7</td>
      <td>17</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>W</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UOL</td>
      <td>Unicorns of Love</td>
      <td>Ahahacik</td>
      <td>GS</td>
      <td>Galatasaray Esports</td>
      <td>Jungle</td>
      <td>Trundle</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>8</td>
      <td>14</td>
      <td>22</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>L</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GS</td>
      <td>Galatasaray Esports</td>
      <td>Mojito</td>
      <td>UOL</td>
      <td>Unicorns of Love</td>
      <td>Jungle</td>
      <td>Talon</td>
      <td>5</td>
      <td>4</td>
      <td>10</td>
      <td>...</td>
      <td>12</td>
      <td>8</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>W</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UOL</td>
      <td>Unicorns of Love</td>
      <td>Nomanz</td>
      <td>GS</td>
      <td>Galatasaray Esports</td>
      <td>Mid</td>
      <td>Leblanc</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>6</td>
      <td>9</td>
      <td>15</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>L</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>


```python
# Let's drop some of the redundant columns or data we don't need
WorldsData.drop(columns = ['Team', 'Opponent', 'Barons Against', 'Dragons Against', 'Herald Against'], inplace = True)

# Then let's add a "KDA" column which represents the ratio between Kills and Assists versus Deaths, a common
WorldsData['KDA'] = np.where(
    WorldsData['Deaths'] == 0,  # This was used to avoid the issue of getting an infinite value when dividing by 0
    round(WorldsData['Kills'] + WorldsData['Assists'], 2),
    round((WorldsData['Kills'] + WorldsData['Assists']) / WorldsData['Deaths'],2))

WorldsData.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team Name</th>
      <th>Player</th>
      <th>Opponent Name</th>
      <th>Position</th>
      <th>Champion</th>
      <th>Kills</th>
      <th>Deaths</th>
      <th>Assists</th>
      <th>Creep Score</th>
      <th>Gold Earned</th>
      <th>Champion Damage Share</th>
      <th>Kill Participation</th>
      <th>Wards Placed</th>
      <th>Wards Destroyed</th>
      <th>Ward Interactions</th>
      <th>Dragons For</th>
      <th>Barons For</th>
      <th>Result</th>
      <th>Herald For</th>
      <th>KDA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Unicorns of Love</td>
      <td>Boss</td>
      <td>Galatasaray Esports</td>
      <td>Top</td>
      <td>Camille</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>188</td>
      <td>11107</td>
      <td>0.17</td>
      <td>0.78</td>
      <td>8</td>
      <td>8</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>L</td>
      <td>1</td>
      <td>1.40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Galatasaray Esports</td>
      <td>Crazy</td>
      <td>Unicorns of Love</td>
      <td>Top</td>
      <td>Gwen</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>217</td>
      <td>12201</td>
      <td>0.20</td>
      <td>0.52</td>
      <td>10</td>
      <td>7</td>
      <td>17</td>
      <td>4</td>
      <td>1</td>
      <td>W</td>
      <td>1</td>
      <td>12.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Unicorns of Love</td>
      <td>Ahahacik</td>
      <td>Galatasaray Esports</td>
      <td>Jungle</td>
      <td>Trundle</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>156</td>
      <td>9048</td>
      <td>0.15</td>
      <td>0.78</td>
      <td>8</td>
      <td>14</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>L</td>
      <td>1</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Galatasaray Esports</td>
      <td>Mojito</td>
      <td>Unicorns of Love</td>
      <td>Jungle</td>
      <td>Talon</td>
      <td>5</td>
      <td>4</td>
      <td>10</td>
      <td>194</td>
      <td>11234</td>
      <td>0.23</td>
      <td>0.65</td>
      <td>12</td>
      <td>8</td>
      <td>20</td>
      <td>4</td>
      <td>1</td>
      <td>W</td>
      <td>1</td>
      <td>3.75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Unicorns of Love</td>
      <td>Nomanz</td>
      <td>Galatasaray Esports</td>
      <td>Mid</td>
      <td>Leblanc</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>216</td>
      <td>9245</td>
      <td>0.29</td>
      <td>0.56</td>
      <td>6</td>
      <td>9</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>L</td>
      <td>1</td>
      <td>1.67</td>
    </tr>
  </tbody>
</table>
</div>


# Exploratory Data Analysis

```python
# Let's take a preliminary look at some of the player's individual stats split based on their position in the game

subset_Worlds = WorldsData.iloc[:,3:14]
sns.pairplot(subset_Worlds, hue="Position")
```




    <seaborn.axisgrid.PairGrid at 0x1fc3fd92af0>




    
![png](README_files/README_6_1.png)
    


Preliminary analysis shows how the support position stands out from the other roles. In particular, they fall short compared to other roles in Kills, Creep Score, Gold Earned and Champion Damage Share but are high in the Wards Placed department. This is namely due to the nature of the role, which doesn't require any "farming" during the "laning phase" of the game and relies more heavily on assisting the team during team fights or strategically placing wards in appropriate locations to provide the team with vision of the map and of the opposing team.

## Factors related to Wins or Losses based on Individual Player Stats

### Creep Score


```python
# Plotting the distribution for Creep Score for players that won vs players that loss their matches
sns.displot(WorldsData, x = "Creep Score", hue = 'Result', kde=True)

# This was also broken down by position given the bimodal distribution see in the first graph
grid = sns.FacetGrid(WorldsData, col = "Position", hue = "Result", col_wrap=5)
grid.map(sns.histplot,"Creep Score")
grid.add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x1fc60aefaf0>




    
![png](README_files/README_9_1.png)
    



    
![png](README_files/README_9_2.png)
    



```python
# Since the difference isn't as apparent visually, let's look at the average creep score table below for comparison
WorldsData[["Result", "Position", "Creep Score"]].groupby(["Result", "Position"]).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Creep Score</th>
    </tr>
    <tr>
      <th>Result</th>
      <th>Position</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">L</th>
      <th>Adc</th>
      <td>273.545455</td>
    </tr>
    <tr>
      <th>Jungle</th>
      <td>172.500000</td>
    </tr>
    <tr>
      <th>Mid</th>
      <td>245.681818</td>
    </tr>
    <tr>
      <th>Support</th>
      <td>32.681818</td>
    </tr>
    <tr>
      <th>Top</th>
      <td>231.863636</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">W</th>
      <th>Adc</th>
      <td>298.454545</td>
    </tr>
    <tr>
      <th>Jungle</th>
      <td>192.954545</td>
    </tr>
    <tr>
      <th>Mid</th>
      <td>274.318182</td>
    </tr>
    <tr>
      <th>Support</th>
      <td>37.181818</td>
    </tr>
    <tr>
      <th>Top</th>
      <td>244.227273</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Ttest to compare the creep scores between players that won and those that loss
# Excluded Supports since they do not CS
ttest_ind(WorldsData[(WorldsData["Result"] == "W") & (WorldsData["Position"] != "Support")]["Creep Score"],
          WorldsData[(WorldsData["Result"] == "L") & (WorldsData["Position"] != "Support")]["Creep Score"],
          alternative='two-sided')
```




    Ttest_indResult(statistic=2.22712091346054, pvalue=0.02722183115902894)



We can see overall that the creep score is slightly higher for players that are winning for positions that are dependent on creep score (Mid, ADC, Jungle and Top). The position with the highest difference in average creep score between the winning and losing team was in Mid followed closely by Adc, then Jungle and lastly Top. We don't see much difference here for Supports since they're role does not rely on last hitting minions/jungle monsters. 

In general our t-test indicates that creep score is a significant factor in determining whether a professional player will win or lose a match for the all positions except for Support.

### Gold Earned


```python
# Plotting the distribution for Gold Earned for players that won vs players that loss their matches
sns.displot(WorldsData, x = "Gold Earned", hue = 'Result', kde=True)

# This was also broken down by position
grid = sns.FacetGrid(WorldsData, col = "Position", hue = "Result", col_wrap=5)
grid.map(sns.histplot,"Gold Earned")
grid.add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x1fc602a5760>




    
![png](README_files/README_14_1.png)
    



    
![png](README_files/README_14_2.png)
    



```python
# Ttest to compare the gold earned between players that won and those that loss
ttest_ind(WorldsData[WorldsData["Result"] == "W"]["Gold Earned"],
          WorldsData[WorldsData["Result"] == "L"]["Gold Earned"],
          alternative='two-sided')
```




    Ttest_indResult(statistic=5.6629794863189575, pvalue=4.664956492986764e-08)



It's quite apparent that gold earned is a clear measure for how well a player is doing. This makes sense, considering gold is the currency earned on all objectives in the game - whether it's kills, assists, destroying enemy wards or towers, damaging enemy champions, or clearing team objectives like dragons.

### KDA: Kills + Assists vs Deaths Ratio


```python
# Plotting the distribution for KDA for players that won vs players that loss their matches
sns.displot(WorldsData, x = "KDA", hue = 'Result', kde=True)

# This was also broken down by position
grid = sns.FacetGrid(WorldsData, col = "Position", hue = "Result", col_wrap=5)
grid.map(sns.histplot,"KDA")
grid.add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x1fc6058a490>




    
![png](README_files/README_18_1.png)
    



    
![png](README_files/README_18_2.png)
    



```python
# Ttest to compare the gold earned between players that won and those that loss
ttest_ind(WorldsData[WorldsData["Result"] == "W"]["KDA"],
          WorldsData[WorldsData["Result"] == "L"]["KDA"],
          alternative='two-sided')
```




    Ttest_indResult(statistic=15.18146889642989, pvalue=5.333364180576791e-36)



KDA seems to show the most distinct difference between the winning and losing team, demonstrating that high kills and assists to deaths ratio is a significant indicator of a player's success in the game. Winning team fights or picking off enemy players swing the game drastically in the team's favor overall. With this we can see that surviving in the game and picking the right fights is very important. 

We'll use this category later on given it's significanct difference between players that won and loss to look at what factors contribute the most to having a high KDA.

## Factors related to Wins or Losses based on Team Objectives

### Heralds
Interestingly enough, the Herald does not seem to have a significant impact on a team's success. The Herald, when used by the team that acquired it, will automatically target the closest enemy tower and deal significant damage to it while it's still alive. A lot of teams strategize using it while other team objectives like the Dragon is up as a means for distraction and splitting the enemy's focus. Given this strategy, it may be more likely that the better timing and use of the Herald has a greater impact as opposed to simply obtaining the objective.


```python
sns.displot(WorldsData, x = "Herald For", hue = 'Result')
```




    <seaborn.axisgrid.FacetGrid at 0x1fc60628e20>




    
![png](README_files/README_22_1.png)
    


### Dragons
Each Dragon acquired provides a permanent team buff. Not surprisingly, acquiring more Dragons (particularly four which leads to a team buff referred to as the "Soul"), leads to higher success in the game.  


```python
sns.displot(WorldsData, x = "Dragons For", hue = 'Result')
```




    <seaborn.axisgrid.FacetGrid at 0x1fc604a2580>




    
![png](README_files/README_24_1.png)
    


### Barons
We see the most distinct difference with the Baron, which gives a temporary team buff that also applies to the minions/creeps that are walking near team that acquires the buff. Teams that were able to obtain this objective were much more likely to win.


```python
sns.displot(WorldsData, x = "Barons For", hue = 'Result')
```




    <seaborn.axisgrid.FacetGrid at 0x1fc608b74f0>




    
![png](README_files/README_26_1.png)
    


# Linear Regression Model for KDA

Given the significant correlation between KDA and whether a player wins their match, I wanted to use a linear regression model to look at what factors had the most impact on KDA.


```python
# First let's do a quick correlation analysis
correlation = WorldsData.corr()

# Creating a half-matrix for our correlation figure
mask = np.triu(np.ones_like(correlation, dtype=bool))

# General dimension for our plot
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(20, 230, as_cmap=True)

# Creating our heatmap for our correlation matrix
sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
```




    <AxesSubplot:>




    
![png](README_files/README_28_1.png)
    


We see that Kills, Deaths and Assists correlate with KDA. But given that KDA is calculated based on these values, for the purpose of the model I decided to leave these out. We can already see that team objectives like the Dragon and Baron seem to show some correlation with KDA.

## First Model
Let's do a linear regression model analysis with all of the factors as a starting off point


```python
# Constant was not used since it decreased the R-squared value by a large margin (was returning R-squared of 0.432)
# Kills, Deaths and Assists were not included since KDA is calculated using these factors

dependent_vars = WorldsData["KDA"]
independent_vars = WorldsData[["Gold Earned", "Creep Score", "Champion Damage Share", "Kill Participation", "Wards Placed", "Wards Destroyed", "Ward Interactions", "Dragons For", "Barons For", "Herald For"]]


dependent_vars = dependent_vars.apply(float)
independent_vars = independent_vars.applymap(float)

lin_reg = sm.OLS(dependent_vars, independent_vars) # Initialize our model based on the ordinarily squared model
reg_results = lin_reg.fit() # Optimizes the fit to the model
print(reg_results.summary()) 
```

                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                    KDA   R-squared (uncentered):                   0.716
    Model:                            OLS   Adj. R-squared (uncentered):              0.704
    Method:                 Least Squares   F-statistic:                              59.23
    Date:                Wed, 15 Dec 2021   Prob (F-statistic):                    6.36e-53
    Time:                        16:37:32   Log-Likelihood:                         -591.60
    No. Observations:                 220   AIC:                                      1201.
    Df Residuals:                     211   BIC:                                      1232.
    Df Model:                           9                                                  
    Covariance Type:            nonrobust                                                  
    =========================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    Gold Earned               0.0003      0.000      1.318      0.189      -0.000       0.001
    Creep Score              -0.0204      0.008     -2.479      0.014      -0.037      -0.004
    Champion Damage Share     4.5291      4.083      1.109      0.269      -3.519      12.577
    Kill Participation        2.5926      1.499      1.730      0.085      -0.362       5.547
    Wards Placed             -0.0332      0.027     -1.246      0.214      -0.086       0.019
    Wards Destroyed          -0.0015      0.040     -0.038      0.970      -0.081       0.077
    Ward Interactions        -0.0347      0.018     -1.905      0.058      -0.071       0.001
    Dragons For               0.7052      0.226      3.118      0.002       0.259       1.151
    Barons For                2.9168      0.592      4.925      0.000       1.749       4.084
    Herald For                1.4386      0.410      3.505      0.001       0.630       2.248
    ==============================================================================
    Omnibus:                       50.507   Durbin-Watson:                   2.211
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               83.551
    Skew:                           1.242   Prob(JB):                     7.20e-19
    Kurtosis:                       4.717   Cond. No.                     5.04e+18
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The smallest eigenvalue is 1.14e-27. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    

We see in the model above that there are a few factors that can be removed due to the high p-values:
- Gold Earned
- Champion Damage Share
- Wards Placed
- Wards Destroyed

## Second Model
Although our R-squared decreases a little, our P-values for each factor looks much better compared to before after dropping those factors.


```python
# We run the model a second time with the above factors removed
independent_vars2 = independent_vars[["Creep Score", "Kill Participation", "Ward Interactions", "Dragons For", "Barons For", "Herald For"]]

dependent_vars = dependent_vars.apply(float)
independent_vars2 = independent_vars2.applymap(float)

lin_reg2 = sm.OLS(dependent_vars, independent_vars2) # Initialize our model based on the ordinarily squared model
reg_results2 = lin_reg2.fit() # Optimizes the fit to the model
print(reg_results2.summary()) 
```

                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                    KDA   R-squared (uncentered):                   0.711
    Model:                            OLS   Adj. R-squared (uncentered):              0.703
    Method:                 Least Squares   F-statistic:                              87.90
    Date:                Wed, 15 Dec 2021   Prob (F-statistic):                    5.44e-55
    Time:                        16:37:32   Log-Likelihood:                         -593.55
    No. Observations:                 220   AIC:                                      1199.
    Df Residuals:                     214   BIC:                                      1219.
    Df Model:                           6                                                  
    Covariance Type:            nonrobust                                                  
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Creep Score           -0.0067      0.002     -2.696      0.008      -0.012      -0.002
    Kill Participation     4.5508      1.103      4.125      0.000       2.376       6.725
    Ward Interactions     -0.0555      0.016     -3.557      0.000      -0.086      -0.025
    Dragons For            0.7693      0.220      3.497      0.001       0.336       1.203
    Barons For             3.3006      0.493      6.699      0.000       2.329       4.272
    Herald For             1.6310      0.381      4.277      0.000       0.879       2.383
    ==============================================================================
    Omnibus:                       52.457   Durbin-Watson:                   2.280
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               88.700
    Skew:                           1.274   Prob(JB):                     5.48e-20
    Kurtosis:                       4.784   Cond. No.                     1.02e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 1.02e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

# Conclusions
From this we can see that overall team objectives seem to play a big role in increasing a player's KDA. Surprisingly, obtaining Heralds also still contribute to high KDA despite the lack of significant difference between winning and losing teams. Overall, this is consistent with current team strategies that heavily revolves around team fights that break out in order to obtain these team objectives. As a result, those that win the fight not only have higher KDAs but also are able to obtain the objectives. This also makes sense considering that high kill participation also plays a big part in KDA.

Interestingly enough, creep score and ward interactions decrease KDA. This suggests that a balance needs to be made in terms of focusing on keeping up a good creep score versus actually attacking the opponent players. Ward interactions also emphasize the importance of good vision on the map and how being seen by the enemy team prior to a fight can put yourself at a disadvantage.
