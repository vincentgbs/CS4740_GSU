import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold

### import data
dataset = pd.read_csv("/home/vincent/CS4740/processed_data.csv")
categorize = {"Winner": {"Red": 1, "Blue": 0, "Draw": -1}}
dataset = dataset[dataset.Winner != -1] ## ignore draws
dataset.replace(categorize, inplace=True)
### cleaning up categories
categories = ["B_current_lose_streak",
"B_current_win_streak",
"B_longest_win_streak",
"B_losses",
"B_total_title_bouts",
"B_win_by_Decision_Majority",
"B_win_by_Decision_Split",
"B_win_by_Decision_Unanimous",
"B_win_by_KO/TKO",
"B_win_by_Submission",
"B_win_by_TKO_Doctor_Stoppage",
"B_wins",
"B_Height_cms",
"B_Reach_cms",
"B_Weight_lbs",
"R_current_lose_streak",
"R_current_win_streak",
"R_longest_win_streak",
"R_losses",
"R_total_title_bouts",
"R_win_by_Decision_Majority",
"R_win_by_Decision_Split",
"R_win_by_Decision_Unanimous",
"R_win_by_KO/TKO",
"R_win_by_Submission",
"R_win_by_TKO_Doctor_Stoppage",
"R_wins",
"R_Height_cms",
"R_Reach_cms",
"R_Weight_lbs",
"Winner"]
dataset.drop(dataset.columns.difference(categories), 1, inplace=True)

##pd.set_option('display.max_rows', 10)
##pd.set_option('display.max_columns', 10)
##pd.set_option('display.width', 500)
##print(dataset.corr(method ='pearson').Winner.values)

##pd.set_option('display.max_rows', 10)
##pd.set_option('display.max_columns', 100)
##pd.set_option('display.width', 1000)
##print(dataset.head)

train, test = train_test_split(dataset, test_size=0.2, random_state=0)
##print(train.Winner.values) ## confirm split
##print(test.Winner.values) ## confirm split
categories.remove("Winner")
X = train[categories]
y = train.Winner

### train logistic regression algorithm
regressor = LogisticRegression()
regressor.fit(X, y)
##print('Intercept: ', regressor.intercept_)
##print('Coefficients: ', regressor.coef_)

### make predictions on test data
y_test = test.Winner.values
y_pred = regressor.predict(test[categories])
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results["Error"] = results.apply(lambda row: (row.Actual-row.Predicted), axis=1)
##print(results)
##print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
##print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
##print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

### adding kfold cross validation to model
scores = []
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    ##print("Train Index: ", train_index, "\n")
    ##print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    regressor.fit(X_train, y_train)
    scores.append(regressor.score(X_test, y_test))

print("kfold scores (k of 10):", scores)
print("Average score of logistic regression:", np.average(scores))
