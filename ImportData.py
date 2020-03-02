import numpy as np
import xlrd
from datetime import datetime
from scipy.linalg import svd
import matplotlib.pyplot as plt

# Load xlsx sheet with data
doc = xlrd.open_workbook('data.xlsx').sheet_by_index(0)

# Save attribute names
attributeNames = doc.row_values(0, 0, 11)
attributeNames2 = doc.row_values(1, 0, 11)
attributeUnits = doc.row_values(2, 0, 11)

# Save clas (season)
classLabels = doc.col_values(9, 3, 333)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(4)))

# Extract vector y
y = np.asarray([classDict[value] for value in classLabels])

# Load data into numpy array
X = np.empty((330, 13))
for i, col_id in enumerate(range(8)):
    X[:, i] = np.asarray(doc.col_values(col_id, 3, 333))

# 1-out-of-k
for c in range(4):
    class_mask = (c==y)
    class_mask2 = (c != y)
    X[class_mask,9+c] = 1
    X[class_mask2,9+c] = 0

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)-1
C = len(classNames)

# Subtract mean value from data
# Normalize / standardize data along columns
X_mean = X.mean(axis=0)
Y = X - np.ones((N, 1))*X_mean
Y_std = np.std(Y,axis=0)
Y = Y / Y_std

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

#U = mat(U)
V = V.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Project data onto principal component space
Z = Y @ V