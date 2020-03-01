# Import data from other script, standardized data matrix is named Y 
from oz_Import import *

plt.figure()
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1*M + m2 + 1)
        for c in range(C2):
            class_mask = (y2==c)
            plt.plot(np.array(Y[class_mask,m2]), np.array(Y[class_mask,m1]), '.')
            if m1==M-1:
                plt.xlabel(attributeNames2[m2], rotation=-45)
            else:
                plt.xticks([])
            if m2==0:
                plt.ylabel(attributeNames2[m1], rotation=0)
            else:
                plt.yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
plt.legend(classNames2)

plt.show()

print('Ran Exercise 4.2.5')