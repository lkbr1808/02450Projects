from mpl_toolkits import mplot3d

# Import data from other script, projected data matrix is named Z
from oz_PCA import *


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