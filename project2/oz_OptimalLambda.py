# exercise 8.1.2

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(1, "Tools/toolbox_02450")
from __init__ import rocplot, confmatplot

from importData import *

font_size = 15
plt.rcParams.update({'font.size': font_size})


K = 30
MR_X_train, MR_X_test, MR_y_train, MR_y_test = train_test_split(Y, y, test_size=(100-100/K)/100, stratify=y)

lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

for k in range(0, len(lambda_interval)):
    mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
    
    mdl.fit(MR_X_train, MR_y_train)

    y_train_est = mdl.predict(MR_X_train).T
    y_test_est = mdl.predict(MR_X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != MR_y_train) / len(MR_y_train)
    test_error_rate[k] = np.sum(y_test_est != MR_y_test) / len(MR_y_test)

    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]

plt.figure(figsize=(8,8))

plt.semilogx(lambda_interval, train_error_rate*100)
plt.semilogx(lambda_interval, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0, 4])
plt.grid()

plt.savefig("project1_figs/fig_OptimalLambdaClass", dpi=300, transparent=True)

plt.show()    
