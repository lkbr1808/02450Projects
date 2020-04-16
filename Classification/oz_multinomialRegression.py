import sklearn.linear_model as lm
import sys
sys.path.insert(1, "Tools/toolbox_02450")
from __init__ import dbplotf, train_neural_net, visualize_decision_boundary
import numpy as np

from importData import *

# Use vandenberg height and humidity
tempX = X[:, [1, 3]]
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

#%% Model fitting and prediction
# Standardize data based on training set
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma



    # Fit multinomial logistic regression model

regularization_strength = 1e-2
    #Try a high strength, e.g. 1e5, especially for synth2, synth3 and synth4
mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                tol=1e-4, random_state=1, 
                                penalty='l2', C=1/regularization_strength)
mdl.fit(X_train,y_train)
y_test_est = mdl.predict(X_test)

test_error_rate = np.sum(y_test_est!=y_test) / len(y_test)

predict = lambda x: np.argmax(mdl.predict_proba(x),1)
plt.figure(2,figsize=(9,9))
visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
plt.title('LogReg decision boundaries')

    # Number of miss-classifications
print('Number of miss-classifications for Multinormal regression:\n\t {0} out of {1}'.format(np.sum(mdl.predict(X_test)!=y_test),len(y_test)))

plt.show()