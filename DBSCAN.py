import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, cluster_optics_dbscan, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

X = pd.read_csv(file_path)
result_file = 'file_path'

# Dropping the CUST_ID column from the data
X = X.drop('id', axis = 1)

# Handling the missing values
#X.fillna(method='ffill', inplace=True)

print(X.head())

pca = PCA(n_components = 2)
X_pc = pca.fit_transform(X)
X_pc = pd.DataFrame(X_pc)
X_pc.columns = ['P1', 'P2']
print(X_pc.head())
del X

kmeans = MiniBatchKMeans(n_clusters=1000,
                        random_state=6,
                        batch_size=128,
                        max_iter=300).fit(X_pc)
#db_default = DBSCAN(eps = 0.02, min_samples = 100, algorithm='ball_tree', metric='haversine').fit(X_pc)
print("DBSCAN Built...")
labels = kmeans.labels_

# Building the label to colour mapping
colours = {}
colours[0] = 'r'
colours[1] = 'g'
colours[2] = 'b'
colours[3] = 'k'
colours[4] = 'y'
colours[5] = 'c'

# Building the colour vector for each data point
cvec = [colours[label] for label in labels]

# For the construction of the legend of the plot
r = plt.scatter(X_pc['P1'], X_pc['P2'], color='r')
g = plt.scatter(X_pc['P1'], X_pc['P2'], color='g')
b = plt.scatter(X_pc['P1'], X_pc['P2'], color='b')
k = plt.scatter(X_pc['P1'], X_pc['P2'], color='k')
y = plt.scatter(X_pc['P1'], X_pc['P2'], color='y')
c = plt.scatter(X_pc['P1'], X_pc['P2'], color='c')

# Plotting P1 on the X-Axis and P2 on the Y-Axis
# according to the colour vector defined
plt.figure(figsize=(20, 20))
plt.scatter(X_pc['P1'], X_pc['P2'], c=cvec)

# Building the legend
plt.legend((r, g, b, k, y, c), ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5'))

plt.show()
