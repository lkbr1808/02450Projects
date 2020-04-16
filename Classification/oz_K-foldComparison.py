from importData import *


# exercise 6.2.1

import sklearn.linear_model as lm
from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier, DistanceMetric

import sys
sys.path.insert(1, "Tools/toolbox_02450")
from __init__ import feature_selector_lr, bmplot


X = Y[:,:M-C]
N, M = X.shape


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Create output lists
Baseline = [None] * K
MR_errorRate = [None] * K
KNN_K = [None] * K
KNN_errorRate = [None] * K
MR_regStrength = [None] * K
MR_errorRate = [None] * K

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    fold_N, fold_M = X_train.shape



    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))

    #Baseline
    

    # Multinomial regression

    K = 20
    MR_X_train, MR_X_test, MR_y_train, MR_y_test = model_selection.train_test_split(X_train, y_train, test_size=.90, stratify=y_train)

    lambda_interval = np.logspace(-8, 2, 50)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))

    for kk in range(0, len(lambda_interval)):
        mdl = lm.LogisticRegression(penalty='l2', C=1/lambda_interval[kk] )
        
        mdl.fit(MR_X_train, MR_y_train)

        y_train_est = mdl.predict(MR_X_train).T
        y_test_est = mdl.predict(MR_X_test).T
        
        train_error_rate[kk] = np.sum(y_train_est != MR_y_train) / len(MR_y_train)
        test_error_rate[kk] = np.sum(y_test_est != MR_y_test) / len(MR_y_test)

        w_est = mdl.coef_[0] 
        coefficient_norm[kk] = np.sqrt(np.sum(w_est**2))

    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]

    MR_regStrength[k] = opt_lambda
    
    regularization_strength = opt_lambda
    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                tol=1e-4, random_state=1, 
                                penalty='l2', C=1/regularization_strength)
    mdl.fit(X_train,y_train)
    y_test_est = mdl.predict(X_test)

    MR_errorRate[k] = np.sum(y_test_est!=y_test)  / len(y_test)

    # K-nearest neighbors

    # Maximum number of neighbors
    KNN_L=20

    KNN_CV = model_selection.LeaveOneOut()
    KNN_errors = np.zeros((fold_N,KNN_L))
    i=0
    for KNN_train_index, KNN_test_index in KNN_CV.split(X_train, y_train):
        print('Crossvalidation fold: %s/%s' % (i+1,fold_N))    
    
        # extract training and test set for current CV fold
        KNN_X_train = X_train[KNN_train_index,:]
        KNN_y_train = y_train[KNN_train_index]
        KNN_X_test = X_train[KNN_test_index,:]
        KNN_y_test = y_train[KNN_test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,KNN_L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l)
            knclassifier.fit(KNN_X_train, KNN_y_train)
            y_est = knclassifier.predict(KNN_X_test)
            KNN_errors[i,l-1] = np.sum(y_est[0]!=KNN_y_test[0])


        i+=1
    
    KNN_errors_temp = KNN_errors.sum(axis=1)

    KNN_K_ = np.argmin(KNN_errors_temp[1:]) + 1
    KNN_K[k] = KNN_K_

    knclassifier = KNeighborsClassifier(n_neighbors=KNN_K_, p=2, 
                                    metric='minkowski',
                                    metric_params={})
    knclassifier.fit(X_train, y_train)
    y_est = knclassifier.predict(X_test)

    KNN_errorRate[k] = np.sum(knclassifier.predict(X_test)!=y_test)  / len(y_test)

    k+=1


# Display results
print('\n')
print(MR_regStrength)
print(MR_errorRate)
print(KNN_K)
print(KNN_errorRate)