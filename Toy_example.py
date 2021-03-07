# EXAMPLE
from sklearn.cluster import KMeans
import numpy as np
from consensusClustering import ConsensusCluster

np.random.seed(0)

# some data
data = np.random.rand(10,3)

# the model
model=ConsensusCluster(cluster=KMeans, L=2, K=10, H=2, resample_proportion=0.5)

# fit the model
model.fit(data)
