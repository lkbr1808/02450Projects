# Import data from other script, standardized data matrix is named Y 
from oz_Import import *

plt.figure(figsize=(12,6))
plt.imshow(Y, interpolation='none', aspect=(10./N), cmap=plt.cm.gray);
plt.xticks(range(10), attributeNames2,rotation=-45)
plt.xlabel('Attributes')
plt.ylabel('Data objects')
plt.title("La Ozone '76 data matrix")
plt.colorbar()

plt.show()