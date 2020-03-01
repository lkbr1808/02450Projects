# Import data from other script, projected data matrix is named Z
from oz_PCA import *

# We saw that we need the first 3 components explaiend more than 80 percent of variance
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r', 'g', 'b']
bw = .2
r = np.arange(1, M+1)
for i in pcs:
    plt.bar(r+i*bw, V[:, i], width=bw)
plt.xticks(r+bw, attributeNames2)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title("LA Ozone '76: PCA Component Coefficients")
plt.show()