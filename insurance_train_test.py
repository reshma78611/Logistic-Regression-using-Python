# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:09:16 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

insu_data=pd.read_csv('C:/Users/HP/Desktop/datasets/claimants.csv')
insu_data.columns
insu_data.drop(['CASENUM'],axis=1,inplace=True)
insu_data.isna().sum()
insu_data.iloc[:,:4]
insu_data.ATTORNEY.value_counts()
insu_data.ATTORNEY.mode()[0]
insu_data.CLMSEX.value_counts()
insu_data.CLMSEX.mode()[0]
#insu_data.ATTORNEY=insu_data.ATTORNEY.fillna(insu_data.ATTORNEY.mode()[0])
#insu_data.ATTORNEY.isna().sum()
#insu_data.CLMSEX=insu_data.CLMSEX.fillna(insu_data.ATTORNEY.mode()[0])
#insu_data.CLMSEX.isna().sum()
insu_data.iloc[:,:4]=insu_data.iloc[:,:4].apply(lambda x:x.fillna(x.mode()[0]))
insu_data.iloc[:,4:]=insu_data.iloc[:,4:].apply(lambda x:x.fillna(x.mean()))
insu_data.isna().sum()


###################Train.Test data#############

from sklearn.model_selection import train_test_split
train,test=train_test_split(insu_data,test_size=0.3,random_state=0)
train.isna().sum();test.isna().sum()

################Model Building##################
##Train data
import statsmodels.formula.api as smf
train_model=smf.logit('ATTORNEY~CLMSEX+CLMINSUR+SEATBELT+CLMAGE+LOSS',data=train).fit()
train_model.summary()
train_pred=train_model.predict(train)
train_pred
train['predicted']=train_pred
train['train_attorney']=np.zeros(938)
train.loc[train_pred>=0.5,'train_attorney']=1
confusion_matrix=pd.crosstab(train.ATTORNEY,train.train_attorney)
confusion_matrix
accuracy_train=(281+378)/(281+180+99+378)
accuracy_train
#70.25
from sklearn.metrics import classification_report
classification_report(train.ATTORNEY,train.train_attorney)
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(train.ATTORNEY,train_pred)
plt.plot(fpr,tpr);plt.xlabel('false positive');plt.ylabel('true positive')

##Test data
test_pred=train_model.predict(test)
test_pred
test['predicted']=test_pred
test['test_attorney']=np.zeros(402)
test.loc[test_pred>=0.5,'test_attorney']=1
test_conf_matrix=pd.crosstab(test.ATTORNEY,test.test_attorney)
test_conf_matrix
test_accuracy=(133+136)/(133+91+42+136)
test_accuracy
#66.91
from sklearn.metrics import classification_report
classification_report(test.ATTORNEY,test.test_attorney)
from sklearn import metrics
fpr_test,tpr_test,threshold=metrics.roc_curve(test.ATTORNEY,test_pred)
plt.plot(fpr_test,tpr_test);plt.xlabel('false positive');plt.ylabel('true positive')
