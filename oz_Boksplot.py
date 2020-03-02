# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from ImportData import *

plt.figure(figsize=(18, 10))
for c in range(C):
    plt.subplot(1, C, c+1)
    class_mask = (y == c)
    plt.boxplot(Y[class_mask, :9])
    plt.title('Class: '+classNames[c])
    plt.xticks(range(1, M-2), attributeNames2[:9], rotation=-45)
    plt.ylim(top=7.2, bottom=-4)

plt.savefig("fig_Boxplot", dpi=500, transparent=True)
