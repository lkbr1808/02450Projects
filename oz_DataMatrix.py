# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from ImportData import *

plt.figure(figsize=(18, 10))

plt.imshow(Y[:, :9], interpolation='none', aspect=(10./N), cmap=plt.cm.gray)
plt.xticks(range(9), attributeNames2, rotation=-45)
plt.xlabel('Attributes')
plt.ylabel('Data objects')
plt.title("La Ozone '76 data matrix")
plt.colorbar()

plt.savefig("fig_DataMatrix", dpi=500, transparent=True)
