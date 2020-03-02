# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from ImportData import *

plt.figure()
for c in range(C):
    plt.subplot(1,C,c+1)
    class_mask = (y==c)
    plt.boxplot(Y[class_mask,:])
    plt.title('Class: '+classNames[c])
    plt.xticks(range(1,M+1),attributeNames2, rotation=-45)

plt.show()