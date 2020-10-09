# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:37:50 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

insurance=pd.read_csv('C:/Users/HP/Desktop/datasets/claimants.csv')
insurance.columns
insurance.drop(['CASENUM'],axis=1,inplace=True)
insurance.columns

insurance.isna().sum()
insurance.iloc[:,:4]
insurance.ATTORNEY.value_counts()
insurance.ATTORNEY.mode()[0]
insurance.CLMSEX.value_counts()
insurance.CLMSEX.mode()[0]

insurance.iloc[:,:4]=insurance.iloc[:,:4].apply(lambda x: x.fillna(x.mode()[0]))
insurance.CLMAGE=insurance.CLMAGE.fillna(insurance.CLMAGE.mean())
insurance.isna().sum()

############Model building###################
import statsmodels.formula.api as smf
insurance_model=smf.logit('ATTORNEY~CLMSEX+CLMINSUR+SEATBELT+CLMAGE+LOSS',data=insurance).fit()
insurance_model.summary()
insurance_model1=smf.logit('ATTORNEY~CLMSEX+CLMINSUR+CLMAGE+LOSS',data=insurance).fit()
insurance_model1.summary()
#seatbelt is not significant
insu_pred=insurance_model1.predict(insurance)
insu_pred
insurance['pred_prob']=insu_pred
insurance['att_val']=0
insurance.loc[insu_pred>=0.5,'att_val']=1
insurance.att_val

#############Confusion matrix############
from sklearn.metrics import classification_report
classification_report(insurance.ATTORNEY,insurance.att_val)
confusion_matrix=pd.crosstab(insurance.ATTORNEY,insurance.att_val)
confusion_matrix
accuracy=(436+504)/(436+249+151+504)
accuracy


############ROC curve##################
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(insurance.ATTORNEY,insu_pred)
plt.plot(fpr,tpr);plt.xlabel('false positive');plt.ylabel('true positive')
roc_auc=metrics.auc(fpr,tpr)#area under curve
roc_auc
