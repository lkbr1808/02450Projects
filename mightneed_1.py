from scipy.linalg import svd
import matplotlib.pyplot as plt

# Import data from other script, data matrix is named X
from script_Import import *


Y = X - np.ones((N,1))*X.mean(axis=0)
Y = Y / np.std(Y,axis=0)
U,S,Vh = svd(Y,full_matrices=False)
V=Vh.T
N,M = X.shape

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2,3,4]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r', 'g', 'b']
bw = .2
r = np.arange(1, M+1)
for i in pcs:
    plt.bar(r+i*bw, V[:, i], width=bw)
plt.xticks(r+bw, attributeNames2)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title("LA Ozone '76: PCA Component Coefficients")
plt.show()

# Inspecting the plot, we see that the 2nd principal component has large
# (in magnitude) coefficients for attributes A, E and H. We can confirm
# this by looking at it's numerical values directly, too:
print('PC2:')
print(V.shape)

# How does this translate to the actual data and its projections?
# Looking at the data for water:

# Projection of water class onto the 2nd principal component.

# all_water_data = Y[y == 4, :]

# print('First water observation')
# print(all_water_data[0, :])

# Based on the coefficients and the attribute values for the observation
# displayed, would you expect the projection onto PC2 to be positive or
# negative - why? Consider *both* the magnitude and sign of *both* the
# coefficient and the attribute!

# You can determine the projection by (remove comments):

# print('...and its projection onto PC2')
# print(all_water_data[0, :]@V[:, 1])

# Try to explain why?