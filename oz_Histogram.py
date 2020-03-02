# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from ImportData import *

plt.figure(figsize=(18, 10))
for m in range(M-4):
    plt.subplot(1, M-4, m+1)
    plt.hist(Y[:, m])
    plt.title(attributeNames2[m])
    plt.ylim(top=120, bottom=0)
    if m != 0:
        plt.yticks([])

plt.suptitle("Histogram", fontsize="x-large")

plt.savefig("figs/fig_Histogram", dpi=300, transparent=True)