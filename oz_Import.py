import numpy as np
import xlrd
from datetime import datetime
from scipy.linalg import svd
import matplotlib.pyplot as plt

# Load xlsx sheet with data
doc = xlrd.open_workbook('data.xlsx').sheet_by_index(0)

# Save attribute names
attributeNames = doc.row_values(0, 0, 11)
attributeNames2 = doc.row_values(1,0,11)
attributeUnits = doc.row_values(2,0,11)

# Save class (month)
classLabels = doc.col_values(10, 3, 333)
classNames = sorted(set(classLabels), key=lambda date: datetime.strptime(date, '%b'))
classDict = dict(zip(classNames, range(1, 13)))

# Save clas (season)
classLabels2 = doc.col_values(11, 3, 333)
classNames2 = sorted(set(classLabels2))
classDict2 = dict(zip(classNames2, range(4)))

# Extract vector y
y = np.asarray([classDict[value] for value in classLabels])
y2 = np.asarray([classDict2[value] for value in classLabels2])

# Load data into numpy array
X = np.empty((330, 10))
for i, col_id in enumerate(range(10)):
    X[:, i] = np.asarray(doc.col_values(col_id, 3, 333))

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)-1
C = len(classNames)
C2 = len(classNames2)

# Subtract mean value from data
# Normalize / standardize data along columns
X_mean = X.mean(axis=0)
Y = X - np.ones((N, 1))*X_mean
Y_std = np.std(Y,axis=0)
Y = Y / Y_std

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

print(X.shape)