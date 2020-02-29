# Import data from other script, standardized data matrix is named Y 
from oz_Import import *

plt.figure()
for c in range(1,C2+1):
    plt.subplot(1,C2,c)
    class_mask = (y==c)
    plt.boxplot(Y[class_mask,:])
    plt.title('Class: '+classNames2[c-1])
    plt.xticks(range(1,M+1),attributeNames2, rotation=-45)

plt.show()

print('Ran Exercise 4.2.4')