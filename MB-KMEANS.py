import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

X = pd.read_csv(file_path)
result_file = 'file_path'

# Dropping the CUST_ID column from the data
X = X.drop('id', axis = 1)

# Handling the missing values
#X.fillna(method='ffill', inplace=True)

print(X.head())

"""
# manually fit on batches
kmeans = MiniBatchKMeans(n_clusters=50,
                         random_state=6,
                          batch_size=64)
kmeans = kmeans.partial_fit(X[0:6,:])
kmeans = kmeans.partial_fit(X[6:12,:])
"""

# fit on the whole data
kmeans = MiniBatchKMeans(n_clusters=1000,
                        random_state=6,
                        batch_size=128,
                        max_iter=300).fit(X)

#kmeans.predict([new_x])


with open(result_file, "a", encoding='utf-8') as f:
    np.savetxt(f, kmeans.labels_, newline="\n")
