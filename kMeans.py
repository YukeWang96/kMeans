#! /usr/bin/python

## This function performs k-means clustering on a dataset

from numpy.random import * 
import numpy as np
import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

def nrow(arr):
    return arr.shape[0]

def ncol(arr):
    return arr.shape[1]


## kwargs could be any of:
## * an integer k representing the desired number of centroids
## * an array object giving initial guesses for the number of centroids
## - Note that input should be done as: kMeans(data, k=3) or
##   kMeans(data, initGuess = np.array([[...],[...],...]))
def kMeans(data, **kwargs):
    
    ## normalize data (make unit variance)
    data = data/data.std()
    
    numFeatures = ncol(data)
    if kwargs.has_key('initGuess'): # is not empty
        ctrds = kwargs['initGuess']
        num_ctrds = nrow(ctrds)
    elif kwargs.has_key('k'): # is not empty
        num_ctrds = kwargs['k']
        ctrds = getRandomCentroids(num_ctrds, numFeatures)
    else:
        raise ValueError('parameter value for kwargs not found; please pass a named value for k or intGuess.')
    if kwargs.get('maxIter'): # is not empty
        try: 
            max_iterations = maxIter
        except ValueError:
            max_iterations = 1000
        except TypeError:
            max_iterations = 1000
    else:
        max_iterations = 1000
    
    if kwargs.get('thresh'): # is not empty
        try: 
            err_thresh = thresh
        except ValueError:
            err_thresh = 1e-6
        except TypeError:
            err_thresh = 1e-6
    else:
        err_thresh = 1e-6
    
    cur_iter = 0
    old_ctrds = None
    while not stopCondition(cur_iter, max_iterations, err_thresh, ctrds, old_ctrds):
        clusters = kMeans_getClusters(data, ctrds)
        
        old_ctrds = ctrds.copy()
        ctrds = kMeans_getCentroids(data, clusters)
        cur_iter+=1
    
    clusterDF = data.copy()
    clusterDF['cluster'] = pd.Series(clusters, index=data.index)
    return (clusterDF, ctrds)

def stopCondition(cur_iter, max_iter, err, ctr1, ctr2):
    ## list of distances between ctr1[0] and ctr2[0], etc. 
    ## (if this ever gives errors, try ctr1.values - ctr2.values)
    if repr(ctr2) == 'None' or repr(ctr1) == 'None':
        return False
    else: 
        sqdiff = np.subtract(ctr1, ctr2)
        sqdiff *= sqdiff
        ctr_dist = sqdiff.sum(1)
        ctr_dist = ctr_dist**.5
        ## ctr_dist = (((ctr1.values-ctr2.values)**2).sum(1))**.5 ## euclidean dist
        return cur_iter >= max_iter or ctr_dist.sum() < err



def kMeans_nearestCentroid(a,ctrds):
    a_diff_sq = (a - ctrds)**2
    local_distortion = (a_diff_sq.sum(1))**.5
    return local_distortion.argmin()



def kMeans_getClusters(data, ctrds):
    return [ kMeans_nearestCentroid(data.iloc[x], ctrds) for x in xrange(nrow(data))]



def kMeans_getCentroids(data, clusters):
    grouped_data = data.groupby(clusters)
    return grouped_data.agg(np.mean)



## Generate centroids for a number of clusters, k, and number of
## dimensions, numFeatures
def getRandomCentroids(k, numFeatures):
    return randn(k, numFeatures)



## Create test data
## need to decide whether dataArr should be DataFrame or ndarray
## dataArr = np.array([[1,2],[3,4],[5,6]])
ranmat1 = randn(10,5)
ranmat2 = randn(10,5) ## + np.array([[2 for _ in range(5)] for _ in range(10)])
df1 = DataFrame(data=ranmat1, columns = list('ABCDE'))
df2 = DataFrame(data=ranmat2, columns = list('ABCDE')).add(2) # separate from 0
dataArr = df1.append(df2)
dataArr = dataArr.set_index([range(len(dataArr))])

## Centroids
mIG_rows = np.random.choice(range(len(dataArr)), size=2, replace=False).tolist()
myInitialGuess = dataArr.iloc[mIG_rows].set_index([range(len(mIG_rows))])


clusterDF, ctrds = kMeans(dataArr, initGuess=myInitialGuess)

print clusterDF
print ctrds

cluscol = []

colorList = ['k', 'r', 'b', 'g', 'm']

for j in clusterDF.cluster:
    cluscol.append(colorList[j])


clusterDF2, ctrds2 = kMeans(dataArr, k=2)
