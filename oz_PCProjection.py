from mpl_toolkits import mplot3d

# Import data from other script, data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V
from oz_Import import *

#U = mat(U)
V = V.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Project data onto principal component space
Z = Y @ V


fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('Projection')
for c in range(4):
    class_mask = (y2 == c)
    #plt.plot(Z[class_mask,0],Z[class_mask,1],'o')
    ax.scatter3D(Z[class_mask,0], Z[class_mask,1], Z[class_mask,2])
plt.legend(classNames2)
plt.xlabel('PC1')
plt.ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()