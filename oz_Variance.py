import matplotlib.pyplot as plt
from scipy.linalg import svd

# Import data from other script, data matrix is named X
from oz_Import import *

# Subtract mean value from data
# Normalize / standardize data along columns
Y = X - np.ones((N, 1))*X.mean(axis=0)
Y = Y / np.std(Y, axis=0)

# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho)+1), rho, 'x-')
plt.plot(range(1, len(rho)+1), np.cumsum(rho), 'o-')
plt.plot([1, len(rho)], [threshold, threshold], 'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative', 'Threshold'])
plt.grid()
plt.show()
