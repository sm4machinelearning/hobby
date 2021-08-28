# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:24:09 2020
@author: Casper Witlox, Adnaan Willson, Gerben van der Schaaf
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text
from sklearn.ensemble import RandomForestClassifier
from MIRCO import MIRCO
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.tree import plot_tree
from matplotlib.pyplot import figure

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

"SET HYPERPARAMETERS HERE"      #values taken from paper teacher
hyps_DT = [8]          #max_dept ONE PARAMETER INVULLEN!      [5, 10, 20]
hyps_RF = [5]          #number of grown trees ONE PARAMETER INVULLEN!   [10, 50, 100]   
num_of_loops = 1        #100

accuracyDT = np.zeros([len(hyps_DT),num_of_loops])
accuracyRF = np.zeros([len(hyps_RF),num_of_loops])
accuracyMIRCO = np.zeros([len(hyps_RF),num_of_loops])
mean_accurDT = [0]*len(hyps_DT)
mean_accurRF = [0]*len(hyps_RF)
mean_accurMIRCO = [0]*len(hyps_RF)

MSE_DT = np.zeros([len(hyps_DT),num_of_loops])
MSE_RF = np.zeros([len(hyps_RF),num_of_loops])
MSE_MIRCO = np.zeros([len(hyps_RF),num_of_loops])
mean_MSE_DT = [0]*len(hyps_DT)
mean_MSE_RF = [0]*len(hyps_RF)
mean_MSE_MIRCO = [0]*len(hyps_RF)

MAE_DT = np.zeros([len(hyps_DT),num_of_loops])
MAE_RF =np.zeros([len(hyps_RF),num_of_loops])
MAE_MIRCO = np.zeros([len(hyps_RF),num_of_loops])
t1 = np.zeros([len(hyps_RF),num_of_loops])
mean_MAE_DT = [0]*len(hyps_DT)
mean_MAE_RF = [0]*len(hyps_RF)
mean_MAE_MIRCO = [0]*len(hyps_RF)

no_rulesDT = np.zeros([len(hyps_DT),num_of_loops])
no_rulesMIRCO = np.zeros([len(hyps_RF),num_of_loops])
mean_no_rulesDT = list([0]*len(hyps_DT))
mean_no_rulesMIRCO = list([0]*len(hyps_RF))

missed_valsMIRCO = np.zeros([len(hyps_RF),num_of_loops])
mean_missed_valsMIRCO = [0]*len(hyps_RF)
mean_t1 = [0]*len(hyps_RF)

no_rulesRF_pertree = np.zeros([len(hyps_RF),num_of_loops])
no_rulesRF_total = np.zeros([len(hyps_RF),num_of_loops])
mean_no_rulesRF_pertree = [0]*len(hyps_RF)
mean_no_rulesRF_total =[0]*len(hyps_RF)

resultsDT = [0]*5
resultsRF = [0]*6
resultsMIRCO = [0]*7

list_of_df_DT = [0]*5
list_of_df_RF = [0]*6
list_of_df_MIRCO = [0]*7

rownames1 = ['Hyperparameter: Maximum depth of tree', 'Average Accuracy Score','Average MSE','Average MAE','Average Number of Rules']
rownames2 = ['Hyperparameter: Number of trees', 'Average Accuracy Score','Average MSE','Average MAE','Average Number of Rules (per tree)','Average Number of Rules (all trees)']
rownames3 = ['Hyperparameter: Number of trees', 'Average Accuracy Score','Average MSE','Average MAE','Average Number of Rules','Average Missed Values','Time Elapsed']

outputDT = pd.DataFrame(np.zeros(5))
outputRF = pd.DataFrame(np.zeros(5))
outputMIRCO = pd.DataFrame(np.zeros(7))

for j in range(0,num_of_loops):
    for i in range(0,len(hyps_DT)):
        "Perform Decision Trees here (classification)"
        DT_classifier = DecisionTreeClassifier(max_depth = hyps_DT[i])
        DT_fit = DT_classifier.fit(X_train,y_train)       
        y_pred_DT = DT_fit.predict(X_test)
        t = (export_text(DT_fit, feature_names = list(X_train_df)))
        for k in t:                         
            if k == '>':
                no_rulesDT[i,j] = no_rulesDT[i,j] + 1
                                
        accuracyDT[i,j] = accuracy_score(y_test,y_pred_DT)
        MSE_DT[i,j] = mean_squared_error(y_test,y_pred_DT)           
        MAE_DT[i,j] = mean_absolute_error(y_test,y_pred_DT)
   


    for i in range(0,len(hyps_RF)):
        """Perform Random Forest here"""          #BELANGRIJK: achteraf input max_depth = hyps_DT[index waarvoor hoogste accuracy bij Decision Trees behaald werd]
        RF_classifier = RandomForestClassifier(max_depth = hyps_DT[i],n_estimators = hyps_RF[i], max_features = "sqrt")
        RF_fit = RF_classifier.fit(X_train,y_train)
        y_pred_RF = RF_fit.predict(X_test)
       
        accuracyRF[i,j] = accuracy_score(y_test,y_pred_RF)
        MSE_RF[i,j] = mean_squared_error(y_test,y_pred_RF)
        MAE_RF[i,j] = mean_absolute_error(y_test,y_pred_RF)
       
        "Perform MIRCO here"
        t0= time.time()
        MIRCO_classifier = MIRCO(RF_fit)
        MIRCO_fit = MIRCO_classifier.fit(X_train,y_train)
        t1[i,j] = time.time() - t0          
        
        y_pred_MIRCO = MIRCO_fit.predict(X_test) 
        accuracyMIRCO[i,j] = accuracy_score(y_test,y_pred_MIRCO)
        MSE_MIRCO[i,j] = mean_squared_error(y_test,y_pred_MIRCO)   
        MAE_MIRCO[i,j] = mean_absolute_error(y_test,y_pred_MIRCO)
        
        no_rulesRF_pertree[i,j] = MIRCO_fit.rf_Rules.sum()/len(MIRCO_fit.rf_Rules)
        no_rulesRF_total[i,j] = MIRCO_fit.rf_Rules.sum() 
        no_rulesMIRCO[i,j] = MIRCO_fit.numOfRules
        missed_valsMIRCO[i,j] = MIRCO_fit.numOfMissed
    

"CALCULATE AVERAGE RESULTS" 
for i in range(0,len(hyps_DT)):               
    mean_accurDT[i] = np.mean(accuracyDT[i,:])
    mean_MSE_DT[i] = np.mean(MSE_DT[i,:])
    mean_MAE_DT[i] = np.mean(MAE_DT[i,:])
    mean_no_rulesDT[i] = np.mean(no_rulesDT[i,:])
    
    resultsDT = [hyps_DT[i],mean_accurDT[i], mean_MSE_DT[i], mean_MAE_DT[i],mean_no_rulesDT[i]]
    list_of_df_DT = pd.DataFrame(resultsDT,index = rownames1)
    
    if i==0:
        outputDT = list_of_df_DT
    else:
        outputDT = pd.concat([outputDT,list_of_df_DT],axis=1)
    

for i in range(0,len(hyps_RF)):
    mean_accurRF[i] = np.mean(accuracyRF[i,:])
    mean_accurMIRCO[i] = np.mean(accuracyMIRCO[i,:])
    
    mean_MSE_RF[i] = np.mean(MSE_RF[i,:])
    mean_MSE_MIRCO[i] = np.mean(MSE_MIRCO[i,:])
    
    mean_MAE_RF[i] = np.mean(MAE_RF[i,:])
    mean_MAE_MIRCO[i] = np.mean(MAE_MIRCO[i,:])
    
    mean_no_rulesRF_pertree = [np.mean(no_rulesRF_pertree[i,:])]
    mean_no_rulesRF_total = [np.mean(no_rulesRF_total[i,:])]
    mean_no_rulesMIRCO[i] = np.mean(no_rulesMIRCO[i,:])

    mean_missed_valsMIRCO = np.mean(missed_valsMIRCO[i,:])  
    
    mean_t1[i] = np.mean(t1[i,:])
    
for i in range(0,len(hyps_RF)):
    resultsRF = [hyps_RF[i],mean_accurRF[i], mean_MSE_RF[i], mean_MAE_RF[i],mean_no_rulesRF_pertree[i],mean_no_rulesRF_total[i]]
    resultsMIRCO = [hyps_RF[i],mean_accurMIRCO[i], mean_MSE_MIRCO[i], mean_MAE_MIRCO[i],mean_no_rulesMIRCO[i], mean_missed_valsMIRCO,mean_t1[i]] 
    list_of_df_RF = pd.DataFrame(resultsRF,index = rownames2)
    list_of_df_MIRCO = pd.DataFrame(resultsMIRCO,index = rownames3)
    
    if i == 0:
        outputRF = list_of_df_RF
        outputMIRCO = list_of_df_MIRCO
    else:
        outputRF = pd.concat([outputRF,list_of_df_RF],axis=1)         
        outputMIRCO = pd.concat([outputMIRCO,list_of_df_MIRCO],axis=1)  
   
"MAKE OUTPUT FOR EXCEL"
writer = pd.ExcelWriter(r'\\solon.prd\files\P\Global\Users\C78579\UserData\Documents\Machine Learning in Finance\Week 3\Assignment (verplicht)\Output_Step1', engine='xlsxwriter')
outputDT.to_excel (writer,sheet_name='Decision Trees',index = True, header=False)
outputRF.to_excel (writer,sheet_name='Random Forest',index = True, header=False)
outputMIRCO.to_excel (writer,sheet_name='MIRCO',index = True, header=False)
writer.save()

"Visualise tree of MIRCO"
fig = figure(figsize=(25,20))
_ = plot_tree(DT_fit, 
                   feature_names=list(X_train_df),
                   filled=True)
fig.savefig("Decision_Tree.png")
