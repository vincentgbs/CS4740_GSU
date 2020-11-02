import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


print("1)")
iris = load_iris()
print("dataset: ")
print(iris.data)
print("shape of dataset: ")
print(iris.data.shape)
print("feature names:")
print(iris.feature_names)
print("targets: ")
print(iris.target)
print("shape of targets")
print(iris.target.shape)

print("2)")
print("convert to DataFrame, ")
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
print("and add column CLASS")
data.insert(5, "CLASS", iris['target'], True)
print(data)

print(data.loc[5:10,'petal length (cm)'])
print(data.loc[7:9,'sepal width (cm)'])

print("mean and standard deviation: ")
print(data['sepal length (cm)'].mean())
print(data['sepal length (cm)'].std())

print(data.describe())

print("3)")
nba = pd.read_csv("/home/vincent/CS4740/nba.csv")

print("missing values in College: ")
print(nba.College.isna().sum())

print(nba.Age.describe())
print("anomalies in Age")
print(nba[nba['Age'].apply(lambda x: not x.isnumeric())])

print("Covariance for Age and Salary: ")
nba['ageNum'] = pd.to_numeric(nba['Age'], errors='coerce')
nba['salNum'] = pd.to_numeric(nba['Salary'], errors='coerce')
print(nba.ageNum.cov(nba.salNum))
plt.matshow(nba.cov())
plt.show()
