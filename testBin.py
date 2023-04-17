# Created by Runhao 
# Revised by Richard on 2/23/2023

from mpi4py import MPI
import sklearn
from sklearn import datasets
from joblib import Memory
from sklearn.datasets import load_svmlight_file
import scipy
import numpy as np
from numpy.random import default_rng
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Need to be changed for each dataset

# numData = 690
# numFeature = 14
numData = 768
numFeature = 8

# numData = 1605
# numFeature = 123


# Partition on features

# f = open('diabetes_scale.txt','r')
f = open('australian_scale.txt','r')
# f = open('a1a.txt','r')
row = np.array([])
col = np.array([])
val = np.array([])
y = np.array([])

offset = numFeature%size
sliceRange = numFeature//size
if (offset == 0):
    l,r = rank*(sliceRange), (rank+1)*(sliceRange)
else:
    if rank < offset:
        l = rank*(sliceRange+1)
        r = l + sliceRange + 1
    else:
        l = offset*(sliceRange+1)+(rank-offset)*sliceRange
        r = l + sliceRange

# counting for row-ptr index
numEntry = 0

for i in range(numData):
    isFirstElementRow = 1
    line = f.readline()
    yElement, xElement = line.split(" ", 1)
    a = int(yElement)
    y = np.append(y, a)

    #Start dealing with features
    featureSplit = xElement.split(" ")
    for j in range(len(featureSplit)-1):
        currCol = featureSplit[j][0]
        currCol = int(currCol)-1
        if currCol >= r:
            break
        if currCol < l:
            continue
        col = np.append(col,currCol)
        if isFirstElementRow == 1:
            row = np.append(row,numEntry)
            isFirstElementRow = 0
        numEntry += 1
        value = featureSplit[j].split(":")[1]
        val = np.append(val, float(value))
row = np.append(row, numEntry)

f.close()

# creating sparse matrix, each node has full y matrix
dataSet = csr_matrix((val, col, row)).toarray()
y = y.reshape(numData,1)

#dataSet = dataSet.transpose()
n, m = dataSet.shape