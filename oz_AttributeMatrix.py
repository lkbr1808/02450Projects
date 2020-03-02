# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from ImportData import *

plt.figure(figsize=(18, 10))
plt.suptitle("Relation between attributes", fontsize = "x-large")

for m1 in range(M-C):
    for m2 in range(M-C):
        plt.subplot(M-C, M-C, m1*(M-C) + m2 + 1)
        for c in range(C):
            class_mask = (y == c)
            plt.plot(np.array(Y[class_mask, m2]),
                     np.array(Y[class_mask, m1]), '.')
            if m1 == M-C-1:
                plt.xlabel(attributeNames2[m2], rotation=-45)
            else:
                plt.xticks([])
            if m2 == 0:
                plt.ylabel(attributeNames2[m1], rotation=0)
            else:
                plt.yticks([])
            # ylim(0,X.max()*1.1)
            # xlim(0,X.max()*1.1)
plt.legend(classNames, bbox_to_anchor=(1.05, 1.11),
           loc='upper left', borderaxespad=0.)

plt.savefig("figs/fig_AttributeMatrix", dpi=300, transparent=True)
