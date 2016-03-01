## My Solution to [*Telstra Network Disruptions*](https://www.kaggle.com/c/telstra-recruiting-network) Kaggle Competition
#### Ranked 15/1010 as team (;´༎ຶД༎ຶ`)

### Feature engineering
As always, feature engineering is the first and the most important step in participating a Kaggle competition. The most important three features are:

1. Treating the location as numeric. The intuition is that records with similar location numbers probably have similar fault severity. It could be that these locations are close to each other and thus their behaviors are similar. Including this feature reduces my mlogloss by ~0.02.
2. Adding the frequency of each location that appears in both training and testing data sets as a feature. This works especially well for high cardinality categorical variables. Including this feature reduces my mlogloss by ~0.01. The variable log feature can be processed following the same idea.

3. The time information. It is very clear from the description of the competition that *The goal of the problem is to predict Telstra network's fault severity at a time at a particular location based on the log data available*. Another interesting finding is that after joining severity_type.csv and the data frame (which is concatenated by train.csv and test.csv) on id, the order of the records contains time information (Check sev_loc.csv in the repo). For each location, the neighboring records tend to have the same fault severity. This implies these records are arranged in the order of time. It is reasonable that the network continues its status for a while and then change to another status. There are two ways to encode this time information. One is for each location, use the row number 
