import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

### START TUTORIAL
### Weather

### import data
##dataset = pd.read_csv("/home/vincent/CS4740/Weather.csv", low_memory=False)

## print(dataset.shape)
## print(dataset.describe())

### scatter plot of min vs max
## dataset.plot(x='MinTemp', y='MaxTemp', style='o')  
## plt.title('MinTemp vs MaxTemp')  
## plt.xlabel('MinTemp')  
## plt.ylabel('MaxTemp')  
## plt.show()

### plot average max temp
## plt.figure(figsize=(15,10))
## plt.tight_layout()
## seabornInstance.distplot(dataset['MaxTemp'])
## plt.show()

### reshaping data to use a single feature
##X = dataset['MinTemp'].values.reshape(-1,1)
##y = dataset['MaxTemp'].values.reshape(-1,1)

### split data into training and test sets
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

###train linear regression algorithm
##regressor = LinearRegression()  
##regressor.fit(X_train, y_train)

###To retrieve the intercept:
##print("int (b): " + str(regressor.intercept_))
###For retrieving the slope:
##print("slope (m): " + str(regressor.coef_))

### make predictions on test data
##y_pred = regressor.predict(X_test)

### check accuracy
##df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
##print(df)

### visualize accuracy
##df1 = df.head(25) # only first 25
##df1.plot(kind='bar',figsize=(16,10))
##plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
##plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
##plt.show()

### plot model with test points
##plt.scatter(X_test, y_test,  color='gray')
##plt.plot(X_test, y_pred, color='red', linewidth=2)
##plt.show()

### evaluate performance of algorithm
##print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
##print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
##print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



### wine quality

### import data
##dataset = pd.read_csv("/home/vincent/CS4740/winequality.csv")

##print(dataset.shape)
##print(dataset.describe())

### check for NaN values
##print(dataset.isnull().any())
### remove any NaN values (none present)
### dataset = dataset.fillna(method='ffill')

### divide data into attributes and labels
##x_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']
##X = dataset[x_columns].values
##y = dataset['quality'].values

### visualize quality
##plt.figure(figsize=(15,10))
##plt.tight_layout()
##seabornInstance.distplot(dataset['quality'])
##plt.show()

### split data into training and testing sets
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### train model
##regressor = LinearRegression() # return numpy array
##regressor.fit(X_train, y_train)

### print coefficients chosen by model
##coeff_df = pd.DataFrame(regressor.coef_, x_columns, columns=['Coefficient'])  
##print(coeff_df)

### run prediction with model
##y_pred = regressor.predict(X_test)

### evaluate performance of model
##df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
##df1 = df.head(25)
##print(df1)

### plot actual vs predicted values
##df1.plot(kind='bar',figsize=(10,8))
##plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
##plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
##plt.show()

### evaluate performance of algorithm
##print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
##print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
##print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

### END OF TUTORIAL



### Homework questions
# a. import into dataframe, b. matplotlib.pyplot (already) imported
wea = pd.read_csv("/home/vincent/CS4740/Weather.csv", low_memory=False)

# c. scatter plot of MinTemp and MaxTemp
plt.scatter(wea.MinTemp, wea.MaxTemp)
plt.show()

# d. These variables appear to have a positive linear relationship.
### That makes sense since the minimum and maximum temperatures for most days should
### be within a reasonably range, so as Min Temp increases for one day, Max Temp will
### likely also increase that day. This is confirmed with a positive ~0.87 correlation. 

# e. print the correlation (0.8783839059497565)
print(
    "Correlation: " + str(wea.MinTemp.corr(wea.MaxTemp))
)

# f. examine WTE column
print(
    wea.WTE.describe()
)
# f. replace all NaN values with 0
wea.WTE.fillna(0, inplace=True)
print(
    wea.WTE.describe()
)
# f. print data.WTE.head()
print(wea.WTE.head())

# g. split the columns into attributes and labels
X = wea['MinTemp'].values.reshape(-1,1) # attributes
y = wea['MaxTemp'].values.reshape(-1,1) # labels
# g. split the data into training and testing data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# h. train a linear regression model
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
# h. print slope and y-intercept
print("y-int (b): " + str(regressor.intercept_))
print("slope (m): " + str(regressor.coef_))

# i. construct Actual and Predicted columns in a new dataframe
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

# j. compute the error and add the Error column
df["Error"] = df.apply(lambda row: (row.Actual-row.Predicted), axis=1)
print(df)

# k. mean of the error? standard deviation of your error?
print("Mean: " + str(df.Error.mean()))
print("Std: " + str(df.Error.std()))

# l. plot scatter plot with model
plt.clf()
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#### m. I asked Professor Camp about this question in class and he said that we should
### plot the same model but against our binned results, so I graphed this. I am honestly
### not sure if the model is more accurate for certain bins than other bins, but I would
### try to justify either argument by looking at which bins (the points) seem to be closest
### to the the model (the line) and which ones are more spread from the model.

# m. compute the mean and std of the error of your model (where a bin is 10 degrees)
bins = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels=[-45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
df['x'] = X_test
df['binnedPredicted'] = pd.cut(df['Predicted'], bins=bins, labels=labels)
df['binnedActual'] = pd.cut(df['Actual'], bins=bins, labels=labels)
# m. display binned results against same model
plt.clf()
plt.scatter(df['x'], df['binnedPredicted'],  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
# m. compute the error for the binned results
df["binnedError"] = df.apply(lambda row: (row.binnedActual-row.binnedPredicted), axis=1)
# m. mean of the error? standard deviation of error?
print("Binned Mean: " + str(df.binnedError.mean()))
print("Binned Std: " + str(df.binnedError.std()))

# Bonus 1. a. through correlation/covariance analysis or visualization
# explain why it may not be best to use all variables when training our model
dataset = pd.read_csv("/home/vincent/CS4740/winequality.csv")
# remove any NaN values (none present)
dataset = dataset.fillna(method='ffill')
### print correlation of all columns
##pd.set_option('display.max_rows', 50)
##pd.set_option('display.max_columns', 50)
##pd.set_option('display.width', 1000)
print(dataset.corr(method ='pearson'))
### a. You can see that some of the input variables have strong correlations.
### Using input variables with strong correlations could lead to duplicate inputs.
### The dependent variable (quality) is likely only dependent on a few attributes,
### so including all of the variables is only introducing extra noise.

# 1. b. train a model different from the tutorial model
# divide data into attributes and labels
x_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']
X = dataset[x_columns].values
y = dataset['quality'].values
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# train model
regressor = LinearRegression() # return numpy array
regressor.fit(X_train, y_train)
# coefficients chosen by model
coeff_df = pd.DataFrame(regressor.coef_, x_columns, columns=['Coefficient'])  
# run prediction with model
y_pred = regressor.predict(X_test)
# evaluate performance of model
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Error': y_test-y_pred})
print("Average Error of Tutorial: " + str(df.Error.mean()))

### Method: Trial and Error: I just toyed with deleting different input variables, focusing
### on the ones that had a lower absolute value correlation with quality,
### until I got a model that appeared to be more accurate than the tutorial's model.
##x_columns = ['fixed acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']
x_columns = ['fixed acidity', 'citric acid', 'chlorides', 'total sulfur dioxide', 'density', 'sulphates','alcohol']
X = dataset[x_columns].values
y = dataset['quality'].values
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# train model
regressor = LinearRegression() # return numpy array
regressor.fit(X_train, y_train)
# coefficients chosen by model
coeff_df = pd.DataFrame(regressor.coef_, x_columns, columns=['Coefficient'])  
# run prediction with model
y_pred = regressor.predict(X_test)
# evaluate performance of model
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Error': y_test-y_pred})
print("Average Error of My Model: " + str(df.Error.mean()))
