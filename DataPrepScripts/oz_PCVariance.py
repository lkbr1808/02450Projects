<<<<<<< HEAD
# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from importData import *

threshold = 0.9

# Plot variance explained
plt.figure(figsize=(18, 10))
plt.plot(range(1, len(rho)+1), rho, 'x-')
plt.plot(range(1, len(rho)+1), np.cumsum(rho), 'o-')
plt.plot([1, len(rho)], [threshold, threshold], 'k--')
plt.title('Variance explained by principal components', fontsize="x-large")
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative', 'Threshold'])
plt.grid()
plt.savefig("project1_figs/fig_PCVariance", dpi=300, transparent=True)
=======
# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from importData import *

threshold = 0.9

# Plot variance explained
plt.figure(figsize=(18, 10))
plt.plot(range(1, len(rho)+1), rho, 'x-')
plt.plot(range(1, len(rho)+1), np.cumsum(rho), 'o-')
plt.plot([1, len(rho)], [threshold, threshold], 'k--')
plt.title('Variance explained by principal components', fontsize="x-large")
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative', 'Threshold'])
plt.grid()
plt.savefig("figs/fig_PCVariance", dpi=300, transparent=True)
>>>>>>> ddbef942b602aab376dc265265d59c1a0cb779dc
