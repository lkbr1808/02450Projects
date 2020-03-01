# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from ImportData import *

plt.figure(figsize=(12,6))
plt.imshow(Y, interpolation='none', aspect=(10./N), cmap=plt.cm.gray);
plt.xticks(range(10), attributeNames2, rotation=-45)
plt.xlabel('Attributes')
plt.ylabel('Data objects')
plt.title("La Ozone '76 data matrix")
plt.colorbar()

plt.show()