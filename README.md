## My Solution to [*Telstra Network Disruptions*](https://www.kaggle.com/c/telstra-recruiting-network) Kaggle Competition
#### Ranked 15/1010 as team (;´༎ຶД༎ຶ`)

### Feature engineering
As always, feature engineering is the first and the most important step in participating a Kaggle competition. The most important three features are:

1. Treating the location as numeric. The intuition is that records with similar location numbers probably have similar fault severity. It could be that these locations are close to each other and thus their behaviors are similar. Including this feature reduces my mlogloss by ~0.02.
2. Adding the frequency of each location in both training and testing data sets as a feature. This works especially well for high cardinality categorical variables. This reduces my mlogloss by ~0.01.

3. The time information. It is very clear from the description of the competition that *The goal of the problem is to predict Telstra network's fault severity at a time at a particular location based on the log data available*.
