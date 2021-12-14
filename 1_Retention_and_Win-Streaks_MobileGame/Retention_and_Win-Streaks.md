# <p align="center">Retention and Win-Streaks in Mobile Game App</p>

<p align="center"><img width="800" height="" src="https://miro.medium.com/max/3200/1*rB-3Q2k7o9qk8IyAzx-TRA.gif"></p>


## Let's get some background!

We've been hired by a mobile gaming company and on the one-year anniversary, we've been tasked with investigating the player retention to learn how our mobile game has performed this past year. The tools we had at our disposal were BigQuery and Google Sheets.

For our investigation we were provided with:
- Match information, including the players who matched against each other, and the outcome
- Player information, including information like the player's age and when they joined


## 30-Day Rolling Retention

First off, we defined retention based on whether or not a player played a game 30 days after joining the game. We wanted to explore the 30-day rolling retention for the past year and see what our percent retention was for each day and whether or not we've seen any growth throughout the year.

### To determine the above, the following steps were taken:

**Step 1: We determined which players had played a game 30 days after joining:**
```sql
WITH player_info_retention_stat AS (
    SELECT 
        DISTINCT p.player_id,
        p.joined,
        IF(
            MAX(day) OVER (PARTITION BY p.player_id) >= joined+30, 
            1, 
            0
        ) AS retention_status,
    FROM `juno-da-bootcamp-project-1.raw_data.player_info` AS p
    LEFT JOIN `juno-da-bootcamp-project-1.raw_data.matches_info` AS m
    ON p.player_id = m.player_id)
```
The above allowed us to create a subquery, called `player_info_retention_stat`, that returns a **1** if the player was retained (their latest game played was after 30 days of joining the game) or a **0** if they were not retained (they did not play a game after 30 days of joining). Please note that we used a `LEFT JOIN` to include even those who downloaded our game but did not end up playing a match against another player as part of our retention stats.

**Step 2: We determined the retention rate and change in retention per day**
```sql
SELECT 
    *,
    CASE 
        WHEN 
        LAG(fractional_retention) OVER (ORDER BY join_day) > 0 
            THEN (fractional_retention - LAG(fractional_retention) OVER (ORDER BY join_day)) / LAG(fractional_retention) OVER (ORDER BY join_day)
        WHEN 
        join_day = 1 
            THEN 0
        ELSE 0
    END AS growth_rate
FROM (    
    SELECT 
        joined AS join_day,
        COUNT(joined) AS total_joined,
        SUM(retention_status) AS total_retained,
        ROUND(SUM(retention_status)/COUNT(joined),4) AS fractional_retention
    FROM player_info_retention_stat AS pr
    GROUP BY joined
    ORDER BY joined)
ORDER BY join_day
```
In the query above, we grouped the player data based on when they joined the game to calculate:
- `total_joined`: The total number of players who joined each day 
- `total_retained`: The total number of players who were retained each day based on the 30-day retention
- `fractional_retention`: The fraction of players that were retained given the total number of players each day

This `fractional_retention` was then used with the `CASE ... WHEN` to calculate the `growth_rate` per day for our final table. This was calculated based on:
```
(fractional_retention current day MINUS fractional_retention of previous day) DIVIDED BY fractional_retention of previous day
```
All of the above allowed us to explore the 30-day rolling retention for the game over the past year.

### 30-Day Rolling Retention Analysis
[Google sheets reference link.](https://docs.google.com/spreadsheets/d/1FC4AcXgUb45kUQqx8Psmrh5OvheLU0R1RB4KC9JC36I/edit?usp=sharing)

<p align="center"><img width="" height="" src="https://github.com/amyw0ng/Juno-College-DA-Bootcamp---Project-1/blob/main/Graphs/Percent%20Retention%20and%20Growth%20Rate%20of%20Retention%20over%20the%20Year%20Graph.png?raw=true"></p>

Given we were working with a 30-day rolling retention, we excluded the last 30 days of the year from our analysis because those who joined in the last 30 days of the year would have automatically been deemed not-retained given we were only working with data up to the end of the year.

From our data above, we saw an average 30-day retention of 65.62% per day for our mobile game over the past year. This was quite consistent for the whole year. This is great considering the mobile app bench mark for 30-day retention sits at 42% - more specifically for mobile games it averages at 27% [^1]. Our mobile app has been performing amazingly well for it's first year in keeping new players engaged for more than 30 days!

In looking at the growth rate, while our retention has been great, we do see that there isn't much change in growth over the course of the year and it has been quite stagnant. The changes we may have implemented this year to the game doesn't seem to have had an impact on the player retention so far so it would be good to explore other incentive systems to increase player engagement. This may also be a sign for us to start directing our energy into long-term retention for those who have stayed with us past the 30-day retention benchmark.


## Effects of Win-Streaks

For our second investigation, we wanted to see if players who were retained after 30 days had a higher win-streak within their first 30 days of joining the game compared to those that were not retained. We wanted to explore if there was a correlation here and whether win-streaks could be a predictor for the 30-day retention.

In order to explore this, we needed to determine the highest win-streak per player based on the games they played within the first 30 days of joining the game. We also wanted to to control for the fact that some players might have played more games within the first 30 days compared to others. We assumed that playing more games could give a higher chance of longer win-streaks by nature, so the total games played per player was taken into account for our analysis.

### To determine the above, the following steps were taken [^2]:

**STEP 1: We identified when a new win-streak happened**
```sql
WITH new_streaks AS (
    SELECT
      player_id,
      day,
      outcome,
      CASE 
            WHEN
            outcome = 'win' AND
            LAG(outcome) OVER (PARTITION BY player_id ORDER BY day) = 'loss' 
                THEN 1 
            WHEN 
            outcome = 'win' AND
            LAG(outcome) OVER (PARTITION BY player_id ORDER BY day) IS NULL
                THEN 1
            ELSE 0 
        END AS new_streak
    FROM (
        SELECT 
            m.player_id,
            m.outcome,
            m.day
        FROM `juno-da-bootcamp-project-1.raw_data.matches_info` m
        JOIN `juno-da-bootcamp-project-1.raw_data.player_info` p
        ON m.player_id = p.player_id
        WHERE 
            m.day <= p.joined+30)),
```
In the subquery above, we only looked at matches that were played within the first 30 days that a player joined the game. We then denoted the start of a win streak with **1** based on if the win was preceded by a loss or if by a null (which would be the first game played). 
 
 **STEP 2: We assigned a unique number to each win-streak per player**
```sql
streak_no_table AS (
    SELECT
        player_id,
        day,
        SUM(new_streak) OVER (PARTITION BY player_id ORDER BY day) streak_no
    FROM new_streaks 
    WHERE
        outcome = 'win'),
```
We then assigned a unique number to each win-streak with the 1's from the query above using the `SUM` function in combination with the `OVER (PARTITION BY...)`.
 
 **STEP 3: We counted the number of wins per streak for each player**
```sql
records_per_streak AS (
    SELECT
        player_id,
        streak_no,
        COUNT(*) AS counter
    FROM streak_no_table
    GROUP BY
        player_id,
        streak_no),
```

**STEP 4: We counted the total number of games each player played within the first 30 days of joining**
```sql
total_games_info AS (
    SELECT 
        m.player_id
        COUNT(*) AS total_games
    FROM `juno-da-bootcamp-project-1.raw_data.matches_info` m
    JOIN `juno-da-bootcamp-project-1.raw_data.player_info` p
    ON m.player_id = p.player_id
    WHERE 
        m.day <= p.joined+30)
    GROUP BY 
        player_id),

```

**STEP 5: Again, we determined whether or not a player was retained based on if they played a game 30 days after joining**
```sql
player_info_retention_stat AS (
    SELECT 
        DISTINCT p.player_id,
        p.joined,
        IF(MAX(day) OVER (PARTITION BY p.player_id) >= joined+30, 1, 0) AS retention_status,
    FROM `juno-da-bootcamp-project-1.raw_data.player_info` p
    LEFT JOIN `juno-da-bootcamp-project-1.raw_data.matches_info` m
    ON p.player_id = m.player_id)
```

**STEP 6: Finally, we found the longest win-streak per player and joined this with the total games played and their retention status**
```sql
SELECT
     records_per_streak.player_id,
     total_games,
     MAX(counter) longest_win_streak,
     retention_status,
 FROM
     records_per_streak 
 JOIN player_info_retention_stat AS pr
 ON records_per_streak.player_id = pr.player_id
 JOIN total_games_info AS tgi
 ON tgi.player_id = records_per_streak.player_id
 GROUP BY 
     records_per_streak.player_id,
     total_games,
     retention_status
```

### Win-Streak Analysis
[Google sheets reference link.](https://docs.google.com/spreadsheets/d/1WXCbHQoklAE41vEg8-a904hEN9RkidIuIRlBNWIxuXc/edit?usp=sharing)

<p align="center"><img width="" height="" src="https://github.com/amyw0ng/Juno-College-DA-Bootcamp---Project-1/blob/main/Graphs/Average%20Highest%20Win-Streak%20Graph.png?raw=true"></p>

We found that the average win-streak within the first 30 days was statistically significantly higher for those that were retained after 30-days, despite having only a difference of 0.41. However, as mentioned before, we wanted to account for the fact that playing more games could naturally inflate win-streaks.

<p align="center"><img width="" height="" src="https://github.com/amyw0ng/Juno-College-DA-Bootcamp---Project-1/blob/main/Graphs/Total%20Games%20Played%20in%20First%2030%20Days%20Graph.png?raw=true"></p>

From our graph above, we can already see that those from the retained group played more games in total within those first 30 days compared to those that were not retained. 

<p align="center"><img width="" height="" src="https://github.com/amyw0ng/Juno-College-DA-Bootcamp---Project-1/blob/main/Graphs/Total%20Games%20Played%20vs.%20Win-Streak%20Graph.png?raw=true"></p>

Once we start comparing the highest average win-streaks relative to the total number of games played, we see a slightly different pattern emerge. First off, we were wary of making conclusions for those that played less than 7 games and those that played more than 22 games in total within the first 30 days. This was because, based on the previous graph, we saw that we had significantly smaller sample sizes for those groups so they were poor representatives. However, if we focused primarily on the 7 to 22 total games range, we could see that there was not much difference between those that were retained versus those that were not retained. This conclusion is further bolstered through a paired t-test which showed that the retained and not retained group were not significantly different. 

From the results above, we can conclude that win-streaks is not a predictor of 30-day retention and doesn't appear to play a part in incentivizing players to keep playing our game. We would probably also want to direct our efforts elsewhere in terms of engagement strategies to improve our 30-day retention.

## Concluding Remarks

Overall, we were able to engineer some new features in our data to explore. While there is much more we could further investigate, we were able to confirm that our mobile game has been performing quite well in engaging new players based on our 30-day retention. We saw some patterns in player behaviour relative to win-streaks that did not appear to correlate with their 30-day retention. This gave us a little insight in guiding us towards other directions in terms of setting up new engagement strategies to further grow our player base.


[^1]: Mobile app 30-day retention industry benchmark - https://www.geckoboard.com/best-practice/kpi-examples/retention-rate/
[^2]: Stackoverflow reference code for win-streak logic - https://stackoverflow.com/questions/17839015/finding-the-longest-streak-of-wins
