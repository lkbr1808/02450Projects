# Import data from other script, data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V
from oz_Import import *

#U = mat(U)
V = V.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Project data onto principal component space
Z = Y @ V

PCs_needed = 5

plt.figure()
for m1 in range(PCs_needed):
    for m2 in range(PCs_needed):
        plt.subplot(PCs_needed, PCs_needed, m1*PCs_needed + m2 + 1)
        for c in range(C2):
            class_mask = (y2==c)
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
plt.legend(classNames2)

plt.show()

print('Ran Exercise 4.2.5')