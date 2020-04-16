# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from importData import *

PCs_needed = 6

PC_list = ["PC%d" % i for i in range(1, PCs_needed+1)]

plt.figure(figsize=(18, 10))

plt.imshow(Z[:, range(PCs_needed)], interpolation='none',
           aspect=(PCs_needed/N), cmap=plt.cm.gray)
plt.xticks(range(PCs_needed), PC_list, rotation=-45)
plt.xlabel('Attributes')
plt.ylabel('Data objects')
plt.title("La Ozone '76 data matrix")
plt.colorbar()

plt.title("Data matrix", fontsize="x-large")

plt.savefig("figs/fig_PCDataMatrix˚", dpi=300, transparent=True)
