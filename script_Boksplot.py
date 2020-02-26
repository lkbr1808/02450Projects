import matplotlib.pyplot as plt

# Import data from other script, data matrix is called X
from script_Import import *


plots = [None] * 9
for i in range(9):
    plots[i] = plt.figure()
    plt.boxplot(X[:,i]) 
    plt.title('Boksplot af %s' % attributeNames[i])

plt.show()
