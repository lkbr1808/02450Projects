<<<<<<< HEAD
# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from importData import *

plt.figure(figsize=(18, 10))

plt.imshow(Y[:, :9], interpolation='none', aspect=(10./N), cmap=plt.cm.gray)
plt.xticks(range(9), attributeNames2, rotation=-45)
plt.xlabel('Attributes')
plt.ylabel('Data objects')
plt.colorbar()

plt.title("Normalised data matrix", fontsize="x-large")

plt.savefig("project1_figs/fig_DataMatrix", dpi=300, transparent=True)
=======
# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from importData import *

plt.figure(figsize=(18, 10))

plt.imshow(Y[:, :9], interpolation='none', aspect=(10./N), cmap=plt.cm.gray)
plt.xticks(range(9), attributeNames2, rotation=-45)
plt.xlabel('Attributes')
plt.ylabel('Data objects')
plt.colorbar()

plt.title("Normalised data matrix", fontsize="x-large")

plt.savefig("figs/fig_DataMatrix", dpi=300, transparent=True)
>>>>>>> ddbef942b602aab376dc265265d59c1a0cb779dc
