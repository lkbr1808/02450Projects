# Import data from other script, standardized data matrix is named Y 
from oz_Import import *

#U = mat(U)
V = V.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Project data onto principal component space
Z = Y @ V

PCs_needed = 5

PC_list = ["PC%d" % i for i in range(PCs_needed)]

plt.figure()
plt.imshow(Z[:,range(PCs_needed)], interpolation='none', aspect=(PCs_needed/N), cmap=plt.cm.gray);
plt.xticks(range(PCs_needed), PC_list, rotation=-45)
plt.xlabel('Attributes')
plt.ylabel('Data objects')
plt.title("La Ozone '76 data matrix")
plt.colorbar()

plt.show()