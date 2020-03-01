# Import data from other script,
# data matrix is named X, standardized and normalized is Y, and components of svd are U,S,V, projection is Z
from ImportData import *

PCs_needed = 5

PC_list = ["PC%d" % i for i in range(PCs_needed)]
Z2 = np.transpose(Z)

colors = ['red','magenta','indigo','blue']

plt.figure()
for c in range(C2):
    class_mask = (y2 == c)
    # plt.subplot(1,PCs_needed, c+1)
    plt.plot(Z2[:, class_mask], color = colors[c], linestyle='dashed')

plt.xticks(range(PCs_needed), PC_list, rotation=-45)
plt.ylabel('Data objects')
plt.title("La Ozone '76 line thing")
plt.legend(['Spring','Summer','Autumn','Winter'])
plt.show()