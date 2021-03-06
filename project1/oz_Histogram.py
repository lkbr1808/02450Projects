<<<<<<< HEAD
# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from importData import *

plt.figure(figsize=(18, 10))
for m in range(M-4):
    plt.subplot(1, M-4, m+1)
    plt.hist(X[:, m])
    plt.title(attributeNames2[m])
    plt.ylim(top=120, bottom=0)
    if m != 0: # Only the first plot will have numbers on y-axis
        plt.yticks([])

plt.suptitle("Histogram", fontsize="x-large")

plt.savefig("project1_figs/fig_Histogram", dpi=300, transparent=True)
=======
# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from importData import *

plt.figure(figsize=(18, 10))
for m in range(M-4):
    plt.subplot(1, M-4, m+1)
    plt.hist(X[:, m])
    plt.title(attributeNames2[m])
    plt.ylim(top=120, bottom=0)
    if m != 0: # Only the first plot will have numbers on y-axis
        plt.yticks([])

plt.suptitle("Histogram", fontsize="x-large")

plt.savefig("figs/fig_Histogram", dpi=300, transparent=True)
>>>>>>> ddbef942b602aab376dc265265d59c1a0cb779dc
