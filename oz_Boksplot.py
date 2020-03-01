# Import data from other script, standardized data matrix is named Y 
from oz_Import import *

plt.figure()
for c in range(C2):
    plt.subplot(1,C2,c+1)
    class_mask = (y2==c)
    plt.boxplot(Y[class_mask,:])
    plt.title('Class: '+classNames2[c])
    plt.xticks(range(1,M+1),attributeNames2, rotation=-45)

plt.show()