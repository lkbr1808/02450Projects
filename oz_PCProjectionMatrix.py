# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from importData import *

PCs_needed = 6

plt.figure(figsize=(18,10))
for m1 in range(PCs_needed):
    for m2 in range(PCs_needed):
        plt.subplot(PCs_needed, PCs_needed, m1*PCs_needed + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plt.plot(np.array(Z[class_mask,m2]), np.array(Z[class_mask,m1]), '.')
            if m1==PCs_needed-1:
                plt.xlabel("PC%d" % (m2+1), rotation=-45)
            else:
                plt.xticks([])
            if m2==0:
                plt.ylabel("PC%d" % (m1+1), rotation=0)
            else:
                plt.yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
plt.suptitle("PCA Scatter plot matrix", fontsize="x-large")
plt.legend(classNames, bbox_to_anchor=(1.05, 1.11), loc='upper left', borderaxespad=0.)


plt.savefig("figs/fig_PCProjectionMatrix", dpi=300, transparent=True)