import matplotlib.pyplot as plt

# Import data from other script, data matrix is called X
from script_Import import *

plots = [None] * 9
for i in range(1):
    plots[i] = plt.figure()
    plt.boxplot(
        X[:,i],  # i'th column of X is used
        sym='x') # symbol for whiskers is set to 'x'
    plt.title('Boksplot af %s' % attributeNames[i])
    plt.ylabel(attributeUnits[i])
    plt.tick_params(
        axis='x',          # changes apply to x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

plt.show()
