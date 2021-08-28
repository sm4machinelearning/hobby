# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:16:51 2020

@author: Casper Witlox, Adnaan Willson, Gerben van der Schaaf
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
#pip install mlxtend
from mlxtend.evaluate import cochrans_q
from sklearn.model_selection import GridSearchCV

"Load data here"
df = pd.read_excel (r'default of credit card clients - preprocessed - 2.xls')
n_total = df.shape[0]
m_total = df.shape[1]

y_total = np.array(df['DEFAULT'])
X_total = np.array(df.loc[:, df.columns != 'DEFAULT'])
m_X = X_total.shape[1]

parameters_1 = {'kernel':('linear','linear','linear','linear','linear','linear' ), 'C':[0.5, 1, 5, 10, 50, 100]}
svc_1 = svm.SVC()
clf_1 = GridSearchCV(svc_1, parameters_1)
clf_1.fit(X_total, y_total)
t_1 = clf_1.cv_results_
print('done svm1')

parameters_2 = {'kernel':('poly', 'poly', 'poly', 'poly', 'poly', 'poly'), 'C':[0.5, 1, 5, 10, 50, 100],'degree': [1, 2, 3, 4, 5, 6], 'gamma': [0.001, 0.0001, 0.0001, 0.01, 0.1, 1]}
svc_2 = svm.SVC()
clf_2 = GridSearchCV(svc_2, parameters_2)
clf_2.fit(X_total, y_total)
t_2 = clf_2.cv_results_

#parameters_3 = {'kernel':('rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf'), 'C':[0.5, 1, 5, 10, 50, 100], 'gamma': [0.001, 0.0001, 0.0001, 0.01, 0.1, 1]}
parameters_3 = {'kernel':('rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf'), 'C':[0.5, 1], 'gamma': [0.001, 0.0001]}
svc_3 = svm.SVC()
clf_3 = GridSearchCV(svc_3, parameters_3)
t_3 = clf_3.cv_results_
clf_3.fit(X_total, y_total)
print('done svm3')

parameters_4 = {'kernel':('sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'), 'C':[0.5, 1, 5, 10, 50, 100], 'gamma': [0.001, 0.0001, 0.0001, 0.01, 0.1, 1]}
svc_4 = svm.SVC()
clf_4 = GridSearchCV(svc_4, parameters_4)
clf_4.fit(X_total, y_total)
t_4 = clf_4.cv_results_
print('done svm4')
"""


#X_fold_total, X_test, y_fold_total, y_test = train_test_split(X_total, y_total, test_size=0.33, random_state=42)


"""



"Set hyperparameters"
C_1 = [0.16, 0.16] 
C_2 = [2.85, 2.85]

C_poly = [0.5, 0.75]
deg_2 = [3, 3]
gam_2 = [1.00, 1.00]

C_3 = [2.51, 2.51]
gam_3 = ['scale', 'scale']

kernel1 = 'linear'
kernel2 = 'poly'     
kernel3 = 'rbf'   

"kfolds Cross-validation" 
kfolds = 5
kf1 = KFold(n_splits=kfolds)
kf2 = KFold(n_splits=kfolds)
kf3 = KFold(n_splits=kfolds)

sum_svm1_1_score = 0
sum_svm1_2_score = 0

sum_svm2_1_score = 0
sum_svm2_2_score = 0

sum_svm3_1_score = 0
sum_svm3_2_score = 0

X_fold_total, X_test, y_fold_total, y_test = train_test_split(X_total, y_total, test_size=0.33, random_state=42)

# LINEAR KERNEL
for train_index, test_index in kf1.split(X_fold_total):
    X_fold_train1, X_fold_val1 = X_fold_total[train_index], X_fold_total[test_index]
    y_fold_train1, y_fold_val1 = y_fold_total[train_index], y_fold_total[test_index]
    
    clf1_1 = svm.SVC(C = C_1[0], kernel = kernel1)
    SVM1_1_fit = clf1_1.fit(X_fold_train1, y_fold_train1)
    svm1_1_score = SVM1_1_fit.score(X_fold_val1, y_fold_val1)
    sum_svm1_1_score += svm1_1_score
    
    clf1_2 = svm.SVC(C = C_1[1], kernel = kernel1)
    SVM1_2_fit = clf1_2.fit(X_fold_train1, y_fold_train1)
    svm1_2_score = SVM1_2_fit.score(X_fold_val1, y_fold_val1)
    sum_svm1_2_score += svm1_2_score
svm1_1_training_score = sum_svm1_1_score / kfolds
svm1_2_training_score = sum_svm1_2_score / kfolds

if svm1_1_training_score > svm1_2_training_score:
    print("Hyperparameter C = ", C_1[0])
    SVM1_fit = clf1_1.fit(X_fold_total, y_fold_total)
    svm1_test_score = SVM1_fit.score(X_test, y_test)
    y_pred_svm1 = SVM1_fit.predict(X_test)
else:
    print("Hyperparameter C = ", C_1[1])
    SVM1_fit = clf1_2.fit(X_fold_total, y_fold_total)
    svm1_test_score = SVM1_fit.score(X_test, y_test)
    y_pred_svm1 = SVM1_fit.predict(X_test)
print("Linear score: ", svm1_test_score)

# POLYNOMIAL KERNEL
for train_index, test_index in kf2.split(X_fold_total):
    X_fold_train2, X_fold_val2 = X_fold_total[train_index], X_fold_total[test_index]
    y_fold_train2, y_fold_val2 = y_fold_total[train_index], y_fold_total[test_index]
    
    clf2_1 = svm.SVC(C = C_2[0], kernel = kernel2, degree = deg_2[0], gamma = gam_2[0], coef0=C_poly[0])
    SVM2_1_fit = clf2_1.fit(X_fold_train2, y_fold_train2)
    svm2_1_score = SVM2_1_fit.score(X_fold_val2, y_fold_val2)
    sum_svm2_1_score += svm2_1_score
        
    clf2_2 = svm.SVC(C = C_2[1], kernel = kernel2, degree = deg_2[1], gamma = gam_2[1], coef0=C_poly[1])
    SVM2_2_fit = clf2_2.fit(X_fold_train2, y_fold_train2)
    svm2_2_score = SVM2_2_fit.score(X_fold_val2, y_fold_val2)
    sum_svm2_2_score += svm2_2_score
svm2_1_training_score = sum_svm2_1_score / kfolds
svm2_2_training_score = sum_svm2_2_score / kfolds

if svm2_1_training_score > svm2_2_training_score:
    print("Hyperparameter C = ", C_2[0])        
    print("Hyperparameter degree = ", deg_2[0])
    print("Hyperparameter gamma = ", gam_2[0])
    print("Hyperparameter constant = ", C_poly[0])
    SVM2_fit = clf2_1.fit(X_fold_total, y_fold_total)
    svm2_test_score = SVM2_fit.score(X_test, y_test)
    y_pred_svm2 = SVM2_fit.predict(X_test)
else:
    print("Hyperparameter C = ", C_2[1])
    print("Hyperparameter degree = ", deg_2[1])
    print("Hyperparameter gamma = ", gam_2[1])    
    print("Hyperparameter constant = ", C_poly[1])
    SVM2_fit = clf2_2.fit(X_fold_total, y_fold_total)
    svm2_test_score = SVM2_fit.score(X_test, y_test)
    y_pred_svm2 = SVM2_fit.predict(X_test)
print("Polynomial score: ", svm2_test_score)



# RADIAL KERNEL
for train_index, test_index in kf3.split(X_fold_total):
    X_fold_train3, X_fold_val3 = X_fold_total[train_index], X_fold_total[test_index]
    y_fold_train3, y_fold_val3 = y_fold_total[train_index], y_fold_total[test_index]
    
    clf3_1 = svm.SVC(C = C_3[0], kernel = kernel3, gamma = gam_3[0])
    SVM3_1_fit = clf3_1.fit(X_fold_train3, y_fold_train3)
    svm3_1_score = SVM3_1_fit.score(X_fold_val3, y_fold_val3)
    sum_svm3_1_score += svm3_1_score
        
    clf3_2 = svm.SVC(C = C_3[1], kernel = kernel3, gamma = gam_3[1])
    SVM3_2_fit = clf3_2.fit(X_fold_train3, y_fold_train3)
    svm3_2_score = SVM3_2_fit.score(X_fold_val3, y_fold_val3)
    sum_svm3_2_score += svm3_2_score
svm3_1_training_score = sum_svm3_1_score / kfolds
svm3_2_training_score = sum_svm3_2_score / kfolds

if svm3_1_training_score > svm3_2_training_score:
    print("Hyperparameter C = ", C_3[0])        
    print("Hyperparameter gamma = ", gam_3[0])
    SVM3_fit = clf3_1.fit(X_fold_total, y_fold_total)
    svm3_test_score = SVM3_fit.score(X_test, y_test)
    y_pred_svm3 = SVM3_fit.predict(X_test)
else:
    print("Hyperparameter C = ", C_3[1])
    print("Hyperparameter gamma = ", gam_3[1])    
    SVM3_fit = clf3_2.fit(X_fold_total, y_fold_total)
    svm3_test_score = SVM3_fit.score(X_test, y_test)
    y_pred_svm3 = SVM3_fit.predict(X_test)
print("Radial score: ", svm3_test_score)

q, p_value = cochrans_q(y_test, 
                        y_pred_svm1, 
                        y_pred_svm2, 
                        y_pred_svm3)

print('Q: %.3f' % q)
print('p-value: %.3f' % p_value)



#CM1_2 = confusion_matrix(y_pred_svm1, y_pred_svm2)
#CM1_3 = confusion_matrix(y_pred_svm1, y_pred_svm3)
#CM2_3 = confusion_matrix(y_pred_svm2, y_pred_svm3)

#print(mcnemar(CM1_2, exact=False))
#print(mcnemar(CM1_3, exact=False))
#print(mcnemar(CM2_3, exact=False))

#MSE
MSE1 = mean_squared_error(y_test,y_pred_svm1)   
MSE2 = mean_squared_error(y_test,y_pred_svm2)   
MSE3 = mean_squared_error(y_test,y_pred_svm3)
print("MSE1: ", MSE1)
print("MSE2: ", MSE2)
print("MSE3: ", MSE3)

"""


