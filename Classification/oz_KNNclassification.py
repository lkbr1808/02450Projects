# exercise 6.3.1

from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix

from importData import *

tempX = X
tempY = y

test_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
test_mask = np.asarray(test_mask)
ix = test_mask.reshape(tempX.shape[0])

X_train = tempX[ix == 0]
X_test = tempX[ix == 1]

y_train = tempY[test_mask == 0]
y_test = tempY[test_mask == 1]


# Plot the training data points (color-coded) and test data points.
plt.figure(1, figsize=(18,10))
styles = ['.b', '.r', '.g', '.y']
for c in range(C):
    class_mask = (y_train==c)
    plt.plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])


# K-nearest neighbors
K=7

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=2
metric = 'minkowski'
metric_params = {} # no parameters needed for minkowski

# You can set the metric argument to 'cosine' to determine the cosine distance
#metric = 'cosine' 
#metric_params = {} # no parameters needed for cosine

# To use a mahalonobis distance, we need to input the covariance matrix, too:
#metric='mahalanobis'
#metric_params={'V': cov(X_train, rowvar=False)}

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)

KNN_errorNum = np.sum(knclassifier.predict(X_test)!=y_test)
KNN_errorRate = KNN_errorNum / len(y_test)
print('Number of miss-classifications for Multinormal regression:\n\t %s out of %s,\nResulting in error rate of:\n\t %s' % (KNN_errorNum, len(y_test), KNN_errorRate))

# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plt.plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
    plt.plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
plt.title('Synthetic data classification - KNN')

plt.savefig("project2_figs/fig_syntheticDataClassification", dpi=300, transparent=True)

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est)
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy
plt.figure(2, figsize=(18,10))
plt.imshow(cm, cmap='binary', interpolation='None')
plt.colorbar()
plt.xticks(range(C)); 
plt.yticks(range(C))
plt.xlabel('Predicted class'); 
plt.ylabel('Actual class')
plt.title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate))

plt.savefig("project2_figs/fig_confusionMatrix", dpi=300, transparent=True)
plt.show()

print("Ran oz_KNNclassification")