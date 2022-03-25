#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from numpy import mean
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold



def importData():
    data = pd.read_csv("Debernardi et al 2020 data.csv")
    return data

def formatData(data):
    #X represents the set of training data seperated from the
    #class label, y
    X_raw = data[["age","plasma_CA19_9","creatinine","LYVE1","REG1B","TFF1","REG1A"]]
    
    S = data[["sex"]].values
    L = len(S)
    for i in range(L):
        if S[i]=='M':
            S[i]=1
        else:
            S[i]=0
    X_raw['sex'] = S
    X = (X_raw-X_raw.min())/ (X_raw.max() - X_raw.min()).values
    y = data[["diagnosis"]].values.ravel()
    L = len(y)
    for i in range(L):
        if y[i]==3:
            y[i]=1
        else:
            y[i]=0
    return X,y

X,y = formatData(importData())

logr = LogisticRegression()
svm = SVC()
rf = RandomForestClassifier(n_estimators=40)

print('Average score for Logistic Regression: \n' + str(mean(cross_val_score(logr, X, y, cv = 10))))
print('Average score for Random Forest Classifier: \n' + str(mean(cross_val_score(rf, X, y, cv = 10))))
print('Average score for Support Vector Machine: \n' + str(mean(cross_val_score(logr, X, y, cv = 10))))

"""
section below commented out as its much less efficient, however it uses
original K split - which is worse so i've wasted my time lol
"""
"""
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

kf = KFold(n_splits=20)
scores_logr = []
scores_rf = []
scores_svm = []
for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y[train], y[test]
    scores_logr.append(get_score(logr, X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(rf, X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(svm, X_train, X_test, y_train, y_test))
print('Scores for Logistic Regression: \n' + str(scores_logr))
print('Scores for Random Forest Classifier: \n' + str(scores_rf))
print('Scores for Support Machine vector: \n' + str(scores_svm))
"""
