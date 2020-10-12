# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:45:17 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bank=pd.read_csv('C:/Users/HP/Desktop/assignments submission/logistic regression/bank-full.csv',sep=';')
bank.columns
bank.isna().sum()



#############Model building#############
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
bank['job']=label_encoder.fit_transform(bank['job'])
bank['marital']=label_encoder.fit_transform(bank['marital'])
bank['education']=label_encoder.fit_transform(bank['education'])
bank['default']=label_encoder.fit_transform(bank['default'])
bank['housing']=label_encoder.fit_transform(bank['housing'])
bank['loan']=label_encoder.fit_transform(bank['loan'])
bank['contact']=label_encoder.fit_transform(bank['contact'])
bank['month']=label_encoder.fit_transform(bank['month'])
bank['poutcome']=label_encoder.fit_transform(bank['poutcome'])
bank['y']=label_encoder.fit_transform(bank['y'])

import statsmodels.formula.api as smf
bank_model=smf.logit('y~age+job+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome',data=bank).fit()
bank_model.summary()
#job is insignificant
bank['job_sq']=bank['job']*bank['job']
bank.drop(['job_sq'],axis=1,inplace=True)
bank_model1=smf.logit('y~age+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome',data=bank).fit()
bank_model1.summary()
bank_pred=bank_model1.predict(bank)
bank_pred
bank['y_pred']=0
bank.loc[bank_pred>=0.5,'y_pred']=1
bank.y_pred

##########confusion matrix################
from sklearn.metrics import classification_report
classification_report(bank['y'],bank['y_pred'])
confusion_matrix=pd.crosstab(bank.y,bank.y_pred)
confusion_matrix
accuracy=(39139+1137)/(39139+783+4152+1137)
accuracy

###############ROC curve#########################
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(bank.y,bank_pred)
plt.plot(fpr,tpr);plt.xlabel('false positive');plt.ylabel('true positive')
roc_auc=metrics.auc(fpr,tpr)#area under curve
roc_auc
