import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# import data
dataset = pd.read_csv("/home/vincent/CS4740/diabetes.csv")

# 1
for attr in dataset.columns:
    print(attr)
    col = dataset[attr]
    print("\t min: " + str(col.min()))
    print("\t max: " + str(col.max()))
    print("\t mean: " + str(col.mean()))
    print("\t median: " + str(col.median()))
    print("\t mode: " + str(col.mode()))
    plt.clf()
    plt.hist(col, bins='auto', rwidth=0.5)
    plt.show()

# 2
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)
print(dataset.corr(method ='pearson'))
#Positive correlations (see comments or excel screenshot in report)

# 3
plt.clf()
corr = dataset.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()

# 4 & 5
print('pregnancies - age')
plt.clf()
axis = dataset.plot.scatter(x='Pregnancies', y='Age', c='Outcome', colormap='plasma')
plt.show(block=axis)
print('glucose - insulin')
plt.clf()
axis = dataset.plot.scatter(x='Glucose', y='Insulin', c='Outcome', colormap='plasma')
plt.show(block=axis)
print('glucose - outcome') ## outcome
plt.clf()
axis = dataset.plot.scatter(x='Glucose', y='Outcome', c='Outcome', colormap='plasma')
plt.show(block=axis)
print('skinthickness - insulin')
plt.clf()
axis = dataset.plot.scatter(x='Insulin', y='SkinThickness', c='Outcome', colormap='plasma')
plt.show(block=axis)
print('skinthickness - bmi')
plt.clf()
axis = dataset.plot.scatter(x='SkinThickness', y='BMI', c='Outcome', colormap='plasma')
plt.show(block=axis)

# 6
train, test = train_test_split(dataset, test_size=0.2, random_state=0)
##print(train.Outcome.values) ## confirm split
##print(test.Outcome.values) ## confirm split
X = train[['Pregnancies','Age', 'Glucose', 'Insulin', 'SkinThickness', 'BMI']]
y = train.Outcome

# 7
##train logistic regression algorithm
regressor = LogisticRegression()
regressor.fit(X, y)
print('Intercept: ', regressor.intercept_)
print('Coefficients: ', regressor.coef_)
### make predictions on test data
y_test = test.Outcome.values
y_pred = regressor.predict(test[['Pregnancies','Age', 'Glucose', 'Insulin', 'SkinThickness', 'BMI']])
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results["Error"] = results.apply(lambda row: (row.Actual-row.Predicted), axis=1)
##print(results)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 8
gnb = GaussianNB()
print('Naive Bayes: ')
y_test = test.Outcome.values
y_pred = gnb.fit(X, y).predict(np.reshape(y_test, (-1, 1)))
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
##print(results)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 9
print('Age')
var = dataset.Age
q25, md, q75 = np.percentile(var, [25, 50, 75])
print("Breakpoints: ", var.min(), ",", q25, ",", md, ",", q75, ",", var.max())
bins = [var.min(), q25, md, q75, var.max()]
plt.clf()
plt.hist(var, bins=bins, rwidth=0.5)
plt.show()

print('Glucose')
var = dataset.Glucose
q25, md, q75 = np.percentile(var, [25, 50, 75])
print("Breakpoints: ", var.min(), ",", q25, ",", md, ",", q75, ",", var.max())
plt.clf()
plt.hist(var, bins=bins, rwidth=0.5)
plt.show()

print('SkinThickness')
var = dataset.SkinThickness
q25, md, q75 = np.percentile(var, [25, 50, 75])
print("Breakpoints: ", var.min(), ",", q25, ",", md, ",", q75, ",", var.max())
plt.clf()
plt.hist(var, bins=bins, rwidth=0.5)
plt.show()

# 10
print("There are ", (dataset['BloodPressure'] == 0).sum(), " empty (0) values for blood pressure.")
dataset['BloodPressure'] = dataset['BloodPressure'].replace(0, dataset.BloodPressure.mean())
print("There are ", (dataset['BloodPressure'] == 0).sum(), " empty (0) values for blood pressure.")
