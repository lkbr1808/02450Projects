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
colors = list(plt.cm.tab10(np.arange(10))) + ["crimson", "indigo"]
ax = plt.axes(projection='3d')
plt.title('Projection')
for c in range(1,13):
    class_mask = (y == c)
    #plt.plot(Z[class_mask,0],Z[class_mask,1],'o')
    ax.scatter3D(Z[class_mask,0], Z[class_mask,1], Z[class_mask,2],color=colors[c-1])
plt.legend(classNames)
plt.xlabel('PC1')
plt.ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()