# EXAMPLE
from sklearn.cluster import KMeans
import numpy as np
from consensusClustering import ConsensusCluster

np.random.seed(0)

# some data
data = np.random.rand(10,5)

# the model
model=ConsensusCluster(cluster=KMeans, L=2, K=3, H=2, resample_proportion=0.5)

# fit the model
model.fit(data)
