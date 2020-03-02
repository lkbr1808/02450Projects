from mpl_toolkits import mplot3d

# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from ImportData import *

plt.figure(figsize=(18, 10))
ax = plt.axes(projection='3d')
plt.title('Projection')
for c in range(C):
    class_mask = (y == c)
    # plt.plot(Z[class_mask,0],Z[class_mask,1],'o')
    ax.scatter3D(Z[class_mask, 0], Z[class_mask, 1], Z[class_mask, 2])
plt.legend(classNames)
plt.xlabel('PC1')
plt.ylabel('PC2')
ax.set_zlabel('PC3')

plt.savefig("fig_PCProjection", dpi=500, transparent=True)
