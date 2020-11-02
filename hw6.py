## Vincent Hu
## Homework 6
## Undergraduate
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA

## 1) a) Run a Z-test on this data
np.random.seed(6)

population_ages1 = stats.poisson.rvs(loc=18, mu=35, size=150000)
population_ages2 = stats.poisson.rvs(loc=18, mu=10, size=100000)
population_ages = np.concatenate((population_ages1, population_ages2))

minnesota_ages1 = stats.poisson.rvs(loc=18, mu=30, size=30)
minnesota_ages2 = stats.poisson.rvs(loc=18, mu=10, size=20)
minnesota_ages = np.concatenate((minnesota_ages1, minnesota_ages2))

print( population_ages.mean() ) ## printed 43.000112
print( minnesota_ages.mean() ) ## printed 39.26

ztest = sm.stats.ztest(minnesota_ages, value=population_ages.mean(), alternative='larger')
print(ztest)
## printed values: (-2.5741944001909505, 0.99497630988083)
## Given that 0.9949 > 0.95 (our 95% confidence interval) we cannot reject the
## null hypothesis that the minnesota mean age is less than or equal to 43.
## If our alternative hypothesis was simply that the minnesota mean age was
## not equal to 43, we could reject that at the 95% confidence level, but our
## alternative hypothesis is specifically that the mean is larger than the
## population mean.

## 1) b) Run a T-test on this data
pct_1 = np.random.random()
pct_2 = np.random.random()

p1 = [pct_1, 1-pct_1]
p2 = [pct_2, 1-pct_2]

pop1 = np.random.choice([0,1], size=(1000,), p=p1)
pop2 = np.random.choice([0,1], size=(1000,), p=p2)

## percentage of children who have had some swimming lessons
print( pop1.mean() ) ## printed 0.831
print( pop2.mean() ) ## printed 0.751

## alternative hypothesis: There is a meaningfull difference between the populations
ttest = sm.stats.ttest_ind(pop2, pop1)
print(ttest)
## printed values: (-4.418839134846804, 1.0454711057342816e-05, 1998.0)
## since the p-value (1.04e-5) is very small, we can reject the null hypothesis
## and say that there is a difference in the percentages of the populations.
## Essentially, this is the "chance" that the true means of these populations
## are the same, given these sample data sets that we have.

## 2) import wine data
file = "/home/vincent/CS4740/winequality5.csv"
df = pd.read_csv(file, names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
   'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'quality'])
df = df[1:] # remove header
df.astype(np.float) # cast values from string to float
## print(df)

## 2) i) split data into train and test data
X = df.drop('quality', axis=1) # attributes
y = df['quality'] # labels
## I ended up moving the train_test_split() function into my function
## in order to randomize the results on 5 different trials.
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
##print("train: ", y_train) ## confirm split
##print("test: ", y_test) ## confirm split

## 2) ii) run 3 different SVMs and compare the performance
## rand is the seed that randomizes the train/test splits
def runSVM(kern, rand=0, deg=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand)
    if (deg):
        svclassifier = SVC(kernel=kern, degree=deg)
    else:
        svclassifier = SVC(kernel=kern)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    print(kern, 'kernel accuracy score:', score)
    return score

linear = []
polynomial = []
gaussian = []

for i in range(0,5):
    linear.append(runSVM('linear', i))
##Results:
##linear kernel accuracy score: 0.6375
##linear kernel accuracy score: 0.596875
##linear kernel accuracy score: 0.565625
##linear kernel accuracy score: 0.584375
##linear kernel accuracy score: 0.559375

for i in range(0,5):
    polynomial.append(runSVM('poly', i, 1))
## reduced the poly argument from 8 to 1 because it took too long
## to run on my computer. If I had a better computer, I would run poly 8
## as the results appear to be slightly better with a larger argument.
    ##Results for poly of 1:
##poly kernel accuracy score: 0.6125
##poly kernel accuracy score: 0.609375
##poly kernel accuracy score: 0.571875
##poly kernel accuracy score: 0.596875
##poly kernel accuracy score: 0.55625
    ##Results for poly of 2:
##poly kernel accuracy score: 0.6375
##poly kernel accuracy score: 0.609375
##poly kernel accuracy score: 0.59375
##poly kernel accuracy score: 0.603125
##poly kernel accuracy score: 0.6125

for i in range(0,5):
    gaussian.append(runSVM('rbf', i))
##Results:
##rbf kernel accuracy score: 0.61875
##rbf kernel accuracy score: 0.609375
##rbf kernel accuracy score: 0.5875
##rbf kernel accuracy score: 0.58125
##rbf kernel accuracy score: 0.625

##print(linear, polynomial, gaussian)
print('linear avg:', np.average(linear))
print('polynomial avg:', np.average(polynomial))
print('gaussian avg:', np.average(gaussian))
## The gaussian kernel appears to perform best (when polynomial is run on 1)
##linear avg: 0.5887499999999999
##polynomial avg: 0.589375
##gaussian avg: 0.6043749999999999

## 2) iii) Run the SVMs with scaled data
## Scale the data (the rest of the SVM code will be the same)
# Get column names first
names = df.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)
##print(scaled_df)

## duplicate the code as above on the scaled data
X = scaled_df.drop('quality', axis=1) # attributes
y = df['quality'] # labels

linear = []
polynomial = []
gaussian = []
for i in range(0,5):
    linear.append(runSVM('linear', i))
for i in range(0,5):
    polynomial.append(runSVM('poly', i, 1))
for i in range(0,5):
    gaussian.append(runSVM('rbf', i))
##linear kernel accuracy score: 0.6
##linear kernel accuracy score: 0.60625
##linear kernel accuracy score: 0.575
##linear kernel accuracy score: 0.58125
##linear kernel accuracy score: 0.559375
##poly kernel accuracy score: 0.5125
##poly kernel accuracy score: 0.509375
##poly kernel accuracy score: 0.446875
##poly kernel accuracy score: 0.515625
##poly kernel accuracy score: 0.540625
##rbf kernel accuracy score: 0.646875
##rbf kernel accuracy score: 0.609375
##rbf kernel accuracy score: 0.609375
##rbf kernel accuracy score: 0.646875
##rbf kernel accuracy score: 0.628125
## Interestingly the raw data performed better than the standardized data.
## Typically standardization is done for a number of reasons, like ease of
## interpretation or using PCA, and generally not about getting a better fit
## in a theoretical sense. So although I did not expect the results to be better
## with the raw data, it is not impossible that raw data could perform better.
## The uses just depend on the situations.

## Bonus) Use PCA on the dataset and then run the SVMs
# run the PCA on the scaled components not raw data
pca = PCA(n_components=5) # first 2 components did not account for enough variance
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents,
    columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])

finalDf = pd.concat([principalDf, df[['quality']]], axis = 1)
##print(finalDf) # display principal components
explainedVariance = pca.explained_variance_ratio_
print(np.sum(explainedVariance))
## I used 5 principal components because that explained about 80% of the variance
## within the original data.
## duplicate the code as above on the scaled data
X = principalDf # attributes (principal components)
y = df['quality'] # labels

linear = []
polynomial = []
gaussian = []
for i in range(0,5):
    linear.append(runSVM('linear', i))
for i in range(0,5):
    polynomial.append(runSVM('poly', i, 8))
for i in range(0,5):
    gaussian.append(runSVM('rbf', i))
##linear kernel accuracy score: 0.615625
##linear kernel accuracy score: 0.60625
##linear kernel accuracy score: 0.546875
##linear kernel accuracy score: 0.56875
##linear kernel accuracy score: 0.553125
    ## poly of 1
##poly kernel accuracy score: 0.615625
##poly kernel accuracy score: 0.60625
##poly kernel accuracy score: 0.55
##poly kernel accuracy score: 0.565625
##poly kernel accuracy score: 0.55625
## PCA allowed the poly kernel to run much faster on my
## computer (one of the purposes of PCA).
    ## poly of 8
##poly kernel accuracy score: 0.51875
##poly kernel accuracy score: 0.525
##poly kernel accuracy score: 0.5
##poly kernel accuracy score: 0.5625
##poly kernel accuracy score: 0.559375
##rbf kernel accuracy score: 0.65625
##rbf kernel accuracy score: 0.621875
##rbf kernel accuracy score: 0.575
##rbf kernel accuracy score: 0.65
##rbf kernel accuracy score: 0.5875
## The PCA did not materially improve the accuracy of the algorithms. This makes sense
## because the PCA only included 80% of the variance in the data. Although 80% is a decent
## amount, 100% is obviously greater than 80%. However, the reduction in attributes allowed
## the kernels to run much faster on my computer, which is one of the purposes of attribute
## reduction, so I still think the PCA was useful.
