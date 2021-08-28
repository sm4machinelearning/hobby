# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:24:09 2020
@author: Casper Witlox, Adnaan Willson, Gerben van der Schaaf
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text
from sklearn.ensemble import RandomForestClassifier
from MIRCO import MIRCO
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import time
import statistics

"LOAD DATA HERE"
df = pd.read_csv('proc_german_num_02 withheader-2.csv')
n_rows = df.shape[0]
n_col = df.shape[1]

y_total = df.iloc[:,[0]]   #credit rating: 1 is qualified for credit, -1 is not qualified for credit
X_total = df.iloc[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.33, random_state=42)

"SET HYPERPARAMETERS HERE"  #values taken from paper teacher
hyps_DT = [1]          #max_dept       [5, 10, 20]  
hyps_RF = [10]         #number of grown trees   [10, 50, 100]   
length_hyps = max(len(hyps_DT),len(hyps_RF))
num_of_loops = 2        #100

accuracyDT = np.zeros([len(hyps_DT),num_of_loops])
accuracyRF = np.zeros([len(hyps_RF),num_of_loops])
accuracyMIRCO = np.zeros([len(hyps_RF),num_of_loops])

MSE_DT = np.zeros([len(hyps_DT),num_of_loops])
MSE_RF = np.zeros([len(hyps_RF),num_of_loops])
MSE_MIRCO = np.zeros([len(hyps_RF),num_of_loops])

MAE_DT = np.zeros([len(hyps_DT),num_of_loops])
MAE_RF =np.zeros([len(hyps_RF),num_of_loops])
MAE_MIRCO = np.zeros([len(hyps_RF),num_of_loops])

no_rulesDT = np.zeros([len(hyps_DT),num_of_loops])
no_rulesRF = np.zeros([len(hyps_RF),num_of_loops])
no_rulesMIRCO = np.zeros([len(hyps_RF),num_of_loops])

missed_valsMIRCO = np.zeros([len(hyps_RF),num_of_loops])
t1 = [0]*len(hyps_RF)

mean_accurDT = [0]*len(hyps_DT)
mean_accurRF = [0]*len(hyps_RF)
mean_accurMIRCO = [0]*len(hyps_RF)

mean_MSE_DT = [0]*len(hyps_DT)
mean_MSE_RF = [0]*len(hyps_RF)
mean_MSE_MIRCO = [0]*len(hyps_RF)

mean_MAE_DT = [0]*len(hyps_DT)
mean_MAE_RF = [0]*len(hyps_RF)
mean_MAE_MIRCO = [0]*len(hyps_RF)

mean_no_rulesDT = [0]*len(hyps_DT)
mean_no_rulesRF = [0]*len(hyps_RF)
mean_no_rulesMIRCO = [0]*len(hyps_RF)

mean_missed_valsMIRCO = [0]*len(hyps_RF)

resultsDT = [0]*5
resultsRF = [0]*5
resultsMIRCO = [0]*7

list_of_df_DT = [0]*5
list_of_df_RF = [0]*5
list_of_df_MIRCO = [0]*7

mean_no_rulesDT = list([0]*len(hyps_DT))
mean_no_rulesRF = list([0]*len(hyps_RF))
mean_no_rulesMIRCO = list([0]*len(hyps_RF))

rownames1 = ['Hyperparameter: Maximum depth of tree', 'Average Accuracy Score','Average MSE','Average MAE','Average Number of Rules']
rownames2 = ['Hyperparameter: Number of trees', 'Average Accuracy Score','Average MSE','Average MAE','Average Number of Rules']
rownames3 = ['Hyperparameter: Number of trees', 'Average Accuracy Score','Average MSE','Average MAE','Average Number of Rules','Average Missed Values','Time Elapsed']

outputDT = pd.DataFrame(np.zeros(5))
outputRF = pd.DataFrame(np.zeros(5))
outputMIRCO = pd.DataFrame(np.zeros(7))
no_rulesRF2 = 0


for j in range(0,num_of_loops):
    for i in range(0,length_hyps):
        "Perform Decision Trees here (classification)"
        DT_classifier = DecisionTreeClassifier(max_depth = hyps_DT[i])
        DT_fit = DT_classifier.fit(X_train,y_train)       
        y_pred_DT = DT_fit.predict(X_test)
        t = (export_text(DT_fit, feature_names = list(X_train)))
        for k in t:                         
            if k == '>':
                no_rulesDT[i,j] = no_rulesDT[i,j] + 1
                                
        accuracyDT[i,j] = accuracy_score(y_test,y_pred_DT)
        MSE_DT[i,j] = mean_squared_error(y_test,y_pred_DT)           
        MAE_DT[i,j] = mean_absolute_error(y_test,y_pred_DT)
       
        """Perform Random Forest here"""
        #use random sample of a number of features (for instance 3 out of 10)
        RF_classifier = RandomForestClassifier(max_depth = hyps_RF[i], max_features = "sqrt")
        RF_fit = RF_classifier.fit(X_train,y_train)
        y_pred_RF = RF_fit.predict(X_test)
       
        accuracyRF[i,j] = accuracy_score(y_test,y_pred_RF)
        MSE_RF[i,j] = mean_squared_error(y_test,y_pred_RF)
        MAE_RF[i,j] = mean_absolute_error(y_test,y_pred_RF)
       
        t0= time.clock()
        "Perform MIRCO here"
        MIRCO_classifier = MIRCO(RF_fit)          
        MIRCO_fit = MIRCO_classifier.fit(X_train.values,y_train.values)
        
        #!!!!!!!!!!!! HIER !!!!!!

        #no_rulesRF[i,j] = MIRCO_fit.initNumOfRules #totaal aantal rules
        no_rulesRF[i ,j] = MIRCO_fit.rf_Rules.sum()/len(MIRCO_fit.rf_Rules) #gemiddeld aantal decision tree rules per iteratie        
        #!!!!!!!!!!!! HIER !!!!!!
           
        no_rulesMIRCO[i,j] = MIRCO_fit.numOfRules
        missed_valsMIRCO[i,j] = MIRCO_fit.numOfMissed
    
        y_pred_MIRCO = MIRCO_fit.predict(X_test.values) 
        accuracyMIRCO[i,j] = accuracy_score(y_test,y_pred_MIRCO)
        MSE_MIRCO[i,j] = mean_squared_error(y_test,y_pred_MIRCO)   
        MAE_MIRCO[i,j] = mean_absolute_error(y_test,y_pred_MIRCO)
        t1[i] = time.clock() - t0
    
"CALCULATE AVERAGE RESULTS" 
for i in range(0,length_hyps):               
    mean_accurDT[i] = np.mean(accuracyDT[i,:])
    mean_accurRF[i] = np.mean(accuracyRF[i,:])
    mean_accurMIRCO[i] = np.mean(accuracyMIRCO[i,:])
    
    mean_MSE_DT[i] = np.mean(MSE_DT[i,:])
    mean_MSE_RF[i] = np.mean(MSE_RF[i,:])
    mean_MSE_MIRCO[i] = np.mean(MSE_MIRCO[i,:])
    
    mean_MAE_DT[i] = np.mean(MAE_DT[i,:])
    mean_MAE_RF[i] = np.mean(MAE_RF[i,:])
    mean_MAE_MIRCO[i] = np.mean(MAE_MIRCO[i,:])
    
    mean_no_rulesDT[i] = np.mean(no_rulesDT[i,:])
    mean_no_rulesRF[i] = np.mean(no_rulesRF[i,:])
    mean_no_rulesMIRCO[i] = np.mean(no_rulesMIRCO[i,:])
    """PAY ATTENTION TO THIS LATER, ERROR DISAPPEARS FOR DEEPER TREES"""
    #mean_missed_valsMIRCO = np.mean(missed_valsMIRCO[i,:])  
    mean_missed_valsMIRCO = [0]*len(hyps_RF)
    
    resultsDT = [hyps_DT[i],mean_accurDT[i], mean_MSE_DT[i], mean_MAE_DT[i],mean_no_rulesDT[i]]
    resultsRF = [hyps_RF[i],mean_accurRF[i], mean_MSE_RF[i], mean_MAE_RF[i],mean_no_rulesRF[i]]
    resultsMIRCO = [hyps_RF[i],mean_accurMIRCO[i], mean_MSE_MIRCO[i], mean_MAE_MIRCO[i],mean_no_rulesMIRCO[i], mean_missed_valsMIRCO[i],t1[i]] 

    list_of_df_DT = pd.DataFrame(resultsDT,index = rownames1)
    list_of_df_RF = pd.DataFrame(resultsRF,index = rownames2)
    list_of_df_MIRCO = pd.DataFrame(resultsMIRCO,index = rownames3)
    if i == 0:
        outputDT = list_of_df_DT
        outputRF = list_of_df_RF
        outputMIRCO = list_of_df_MIRCO
    else:
        outputDT = pd.concat([outputDT,list_of_df_DT],axis=1)     
        outputRF = pd.concat([outputRF,list_of_df_RF],axis=1)
        outputMIRCO = pd.concat([outputMIRCO,list_of_df_MIRCO],axis=1)        
   
"MAKE OUTPUT FOR EXCEL"
writer = pd.ExcelWriter(r'\\solon.prd\files\P\Global\Users\C78579\UserData\Documents\Machine Learning in Finance\Week 3\Assignment (verplicht)\Output_Step1.xlsx', engine='xlsxwriter')
outputDT.to_excel (writer,sheet_name='Decision Trees',index = True, header=False)
outputRF.to_excel (writer,sheet_name='Random Forest',index = True, header=False)
outputMIRCO.to_excel (writer,sheet_name='MIRCO',index = True, header=False)
writer.save()