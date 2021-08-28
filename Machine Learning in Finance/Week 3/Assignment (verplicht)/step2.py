# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 10:30:00 2020

@author: Casper Witlox, Gerben van der Schaaff and Adnaan Willson
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from quickMIRCO import MIRCO
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import time

"""
IMPORTANT, first import the output file from Step1 to obtain the average accuracy of MIRCO / Run step1.py
"""

df_step1 = pd.read_excel(r'\\solon.prd\files\P\Global\Users\C78579\UserData\Documents\Machine Learning in Finance\Week 3\Assignment (verplicht)\Output_Step1',sheet_name='MIRCO',header=None,index_col=0)
mean_acc_step1 = df_step1.loc['Average Accuracy Score']
mean_MSE_step1 = df_step1.loc['Average MSE']
mean_MAE_step1 = df_step1.loc['Average MAE']
time_step1_perhyps_RF = df_step1.loc['Time Elapsed']

"LOAD DATA HERE"
df = pd.read_csv('german.data-numeric.csv',sep=',', header=None)
n_rows = df.shape[0]
n_col = df.shape[1]

y_total = np.array(df[24]) 
X_total = np.array(df.iloc[:,0:24])

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(X_total):
    X_train, X_test = X_total[train_index], X_total[test_index]
    y_train, y_test = y_total[train_index], y_total[test_index]

X_train_df = pd.DataFrame(X_train)

hyps_DT = [8]      
hyps_RF = [5]
num_of_loops = 100                      

accuracy_step2 = np.zeros([len(hyps_RF),num_of_loops])
MSE_step2 = np.zeros([len(hyps_RF),num_of_loops])
MAE_step2 = np.zeros([len(hyps_RF),num_of_loops])

t2 = np.zeros([len(hyps_RF),num_of_loops])
mean_t2 = [0]*len(hyps_RF)


mean_accur_step2 = [0]*len(hyps_RF)
mean_MSE_step2 = [0]*len(hyps_RF)
mean_MAE_step2 = [0]*len(hyps_RF)

results_step2 = [0]*9
list_of_df_step2 = [0]*9
output_step2 = pd.DataFrame(np.zeros(9))

for j in range(0,num_of_loops):
    for i in range(0,len(hyps_RF)):         
        "Random Forest needed to perform MIRCO"             
        RF_classifier = RandomForestClassifier(max_depth = hyps_DT[i], n_estimators = hyps_RF[i], max_features = "sqrt") #KEYERROR when max_depth = hyps_RF[i]
        RF_fit = RF_classifier.fit(X_train,y_train)
        
        "Perform MIRCO here"
        t0= time.time()
        MIRCO_classifier = MIRCO(RF_fit)  
        MIRCO_fit = MIRCO_classifier.fit(X_train,y_train)
        t2[i,j] = time.time() - t0                  #imported from quickMIRCO.py, not MIRCO.py


        y_pred_step2 = MIRCO_fit.predict(X_test) 
        accuracy_step2[i,j] = accuracy_score(y_test,y_pred_step2)
        MSE_step2[i,j] = mean_squared_error(y_test,y_pred_step2)
        MAE_step2[i,j] = mean_absolute_error(y_test,y_pred_step2)

for i in range(0,len(hyps_RF)):               
    mean_accur_step2[i] = np.mean(accuracy_step2[i,:])
    mean_MSE_step2[i] = np.mean(MSE_step2[i,:])
    mean_MAE_step2[i] = np.mean(MAE_step2[i,:])
    
    mean_acc_step1 = pd.DataFrame(mean_acc_step1)
    mean_MSE_step1 = pd.DataFrame(mean_MSE_step1)
    mean_MAE_step1 = pd.DataFrame(mean_MAE_step1)
    
    time_step1_perhyps_RF = pd.DataFrame(time_step1_perhyps_RF)
    
    mean_t2[i] = np.mean(t2[i,:])
    
    results_step2 = [hyps_RF[i],mean_accur_step2[i], mean_MSE_step2[i], mean_MAE_step2[i],mean_t2[i],mean_acc_step1.values[i][i], mean_MSE_step1.values[i][i], mean_MAE_step1.values[i][i],time_step1_perhyps_RF.values[i][i]]
    list_of_df_step2 = pd.DataFrame(results_step2,index = ['Hyperparameter: Number of trees', 'Average Accuracy Score Step 2','Average MSE Step 2','Average MAE Step 2', 'Time Elapsed Step 2','Average Accuracy Score Step 1','Average MSE Step 1','Average MAE Step 1','Time Elapsed Step 1'])
    
    if i == 0:
        output_step2 = list_of_df_step2
    else:
        output_step2 = pd.concat([output_step2,list_of_df_step2],axis=1)        
   
"MAKE OUTPUT FOR EXCEL"
writer = pd.ExcelWriter(r'\\solon.prd\files\P\Global\Users\C78579\UserData\Documents\Machine Learning in Finance\Week 3\Assignment (verplicht)\Output_Step2', engine='xlsxwriter')
output_step2.to_excel (writer,sheet_name='MIRCO',index = True, header=False)
writer.save()
