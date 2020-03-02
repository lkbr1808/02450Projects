# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from ImportData import *

PCs_needed = 6

legendStrs = ['PC'+str(e+1) for e in range(PCs_needed)]
bw = .12
r = np.arange(1, M+1)
for i in range(PCs_needed):
    plt.bar(r+i*bw, V[:, i], width=bw)
plt.xticks(r+bw, attributeNames2)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)

plt.title("LA Ozone '76: PCA Component Coefficients")
plt.show()