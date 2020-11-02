import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch

# import data
file = "/home/vincent/CS4740/winequality5.csv"
df = pd.read_csv(file, names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
   'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'quality'])
df = df[1:] # remove header
df.astype(np.float) # cast values from string to float

### Scaling the features before applying PCA (PCA is affected by scale)
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['quality']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

# 7. Perform Principal Component Analysis on the given dataset
pca = PCA(n_components=4) # first 2 components did not account for enough variance
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
    columns = ['principal component 1', 'principal component 2', 'pc3', 'pc4'])

finalDf = pd.concat([principalDf, df[['quality']]], axis = 1)
#print(finalDf) # display

# 7. Print the explained variance ratios for all principal components
explainedVariance = pca.explained_variance_ratio_
print(explainedVariance)
#[0.28173931 0.1750827  0.1409585  0.11029387]
### Most of the data is contained in the first few principal components.
### 28.17% of the variance is contained in the first component and slightly
### less is in each subsequent component as displayed above.
### The first 4 principle components account for ~70% of the variance.

# bar plot of the explained variance for each principle component
label = ['principal component 1', 'principal component 2', 'pc3', 'pc4']
def plot_bar_pc():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, explainedVariance)
    plt.xlabel('component', fontsize=5)
    plt.ylabel('explained variance', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Explained variance of each principle component')
    plt.show()
plot_bar_pc()

# 8 & 9. Make a scatter plot of the data using the first 2 principal components
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['4', '5', '6', '7']
# 9. Update your plot to include color-coded information about the classes
colors = ['r', 'g', 'b', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['quality'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
        finalDf.loc[indicesToKeep, 'principal component 2'],
        c = color,
        s = 50)
ax.legend(targets)
plt.show()

# 10. Make a 3D scatter plot of the data using the first 3 principal components
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(finalDf['principal component 1'],
    finalDf['principal component 2'],
    finalDf['pc3'])
pyplot.show()

# 11. Perform K-means clustering on the first 2 principal components
def kmeans(numberOfK):
    kmeans = KMeans(n_clusters=numberOfK, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(x)
    plt.clf()
    plt.scatter(x[:,0], x[:,1])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()

# 11. Use k-values of 1-10
for k in range(1, 11):
    kmeans(k)

# 11. Plot the resulting inertia graph
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Inertia graph (wcss)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
### It is hard to identify a clear elbow point from the graph, but my best guess is ~7.
### Below (in number 12) I will use 7 as the k value. There does not appear to be a relationship
### between the value of k and the distinct values for the quality target attribute.
### However, since the elbow point is not clear, there could easily be a relationship, given that
### the values between 4 and 7 all have the potential to be elbow points.

# 12. Plot the first 2 Principle Components (color coded)
kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(x)
plt.clf()
plt.scatter(x[:, 0], x[:, 1], c=pred_y, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

# 13. Perform Hierarchical Clustering on the first 2 principal components
# Reduce the dataset into 100 randomly selected rows
X = df.sample(n = 100)

# Plot the resulting dendogram
plt.clf()
dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
plt.title('Dendrogram')
plt.show()
### The dendrogram is showing the breakdown/accumulation of each of the data points.
### Starting at the bottom, each data point is its own cluster and they are grouped
### together with the most similar data points and slowly accumulate into clusters
### until the data is agglomerated at the top.
