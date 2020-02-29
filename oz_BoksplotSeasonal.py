# requires data from exercise 4.1.1
from oz_Import import *

plt.figure()
for c in range(1,C2+1):
    plt.subplot(1,C2,c)
    class_mask = (y==c) # binary mask to extract elements of class c
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c
    
    plt.boxplot(Y[class_mask,:])
    #title('Class: {0}'.format(classNames[c]))
    plt.title('Class: '+classNames2[c-1])
    plt.xticks(range(1,M+1),attributeNames2, rotation=-45)

plt.show()

print('Ran Exercise 4.2.4')