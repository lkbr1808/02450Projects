# Import data from other script, data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V
from oz_Import import *

# Draw the plots
plt.boxplot(Y)
r = np.arange(1, X.shape[1]+1)
plt.xticks(r, attributeNames2)
plt.show()

## Uncomment below for individual, unstandardized boxplots. Chance X to Y for standardized
# plots = [None] * 9
# for i in range(9):
#     plots[i] = plt.figure()
#     plt.boxplot(
#         X[:,i],  # i'th column of X is used
#         sym='x') # symbol for outliers is set to 'x'
#     plt.title('Boksplot af %s' % attributeNames[i])
#     plt.ylabel(attributeUnits[i])
#     plt.tick_params(
#         axis='x',          # changes apply to x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         labelbottom=False) # labels along the bottom edge are off

# plt.show()