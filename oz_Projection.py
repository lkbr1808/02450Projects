import matplotlib.pyplot as plt
from scipy.linalg import svd

# Import data from other script, data matrix is called X
from oz_Import import *

# Subtract mean value from data
# Normalize / standardize data along columns
Xc = X - np.ones((N, 1))*X.mean(axis=0)
Xc = Xc / np.std(Xc, axis=0)

# Number of principal components for reconstruction
K = 3

# PCA by computing SVD of Y
U,S,V = svd(Xc,full_matrices=False)

#U = mat(U)
V = V.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Project data onto principal component space
Z = Xc @ V

f = plt.figure()
plt.title('Projection')
for c in range(12):
    class_mask = (y == c)
    plt.plot(Z[class_mask,0], Z[class_mask,1], 'o')
plt.legend(classNames)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()