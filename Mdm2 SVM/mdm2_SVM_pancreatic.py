#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:55:26 2022

@author: jamiethorn
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix


def importData():
    data = pd.read_csv("Debernardi et al 2020 data.csv")
    data['sex'] = data['sex'].map({'F': 1, 'M': 0})
    return data
    
def formatData(data):
    #X represents the set of training data seperated from the
    #class label, y 
    X_raw = data[["sample_id","age","creatinine","LYVE1","REG1B","TFF1"]]
    X_raw.set_index("sample_id", inplace=True)
    y = data[["sample_id","diagnosis"]]
    y.set_index("sample_id", inplace=True)
    y.replace(to_replace = [1,2], value = 0, inplace = True)
    y.replace(to_replace = 3, value = 1,  inplace = True)
    return X_raw,y
def normalise(X_raw):
    return (X_raw-X_raw.min())/ (X_raw.max() - X_raw.min())
def pca(X,y,data):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    principalDf["sample_id"] = data[["sample_id"]]
    principalDf.set_index("sample_id", inplace=True)
    finalDf = pd.concat([principalDf, y ], axis = 1)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [1, 0]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['diagnosis'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()                


def t_sne(X,y,data):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=800)
    tsne_results = tsne.fit_transform(X)
    df = pd.DataFrame(data = tsne_results, columns = ['tsne comp 1', 'tsne comp 2'])
    df["sample_id"] = data[["sample_id"]]
    df.set_index("sample_id", inplace=True)
    finalDf = pd.concat([df, y], axis = 1)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('tsne comp 1', fontsize = 15)
    ax.set_ylabel('tsne comp 2', fontsize = 15)
    ax.set_title('2 component t-SNE', fontsize = 20)
    targets = [1, 0]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['diagnosis'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'tsne comp 1']
                   , finalDf.loc[indicesToKeep, 'tsne comp 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()   
    
def crossVal(X,y,svm):
    
    cv = LeaveOneOut()
    scores = cross_val_score(svm, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

def svm(X,y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.50, test_size=0.50)
    rbf = SVC(kernel='rbf', gamma=0.1, C=1,probability=True, class_weight='balanced').fit(X_train, y_train)
    rbf_penalised = SVC(kernel='rbf', gamma='scale', C=1,probability=True ).fit(X_train, y_train)
    rbf_pred = rbf.predict(X_test)
    rbf_accuracy = accuracy_score(y_test, rbf_pred)
    rbf_probabilities = rbf.predict_proba(X_test)
    rbf_confidence_score = []
    print(X_train)
    for each in rbf_probabilities:
        #negative score denotes more confidence in not cancerous and vice versa
        score = each[1]-each[0]
        rbf_confidence_score.append(score)
    print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    rbf_results = pd.DataFrame(data = rbf_pred, columns = ['results'])
    rbf_results['sample_id'] = y_test.index
    rbf_results.set_index('sample_id', inplace = True)
    rbf_results['confidence'] = rbf_confidence_score
    crossVal(X,y,rbf)
    print(confusion_matrix(y_test,rbf_pred))
    return y_test,rbf_results

def typeAnalysis(expected, results, data):
    data.set_index("sample_id", inplace = True)
    ids = list(expected.index)
    comparison = pd.concat([expected, results], axis = 1)
    testpatients = data.loc[ids]
    comparison['type'] = testpatients['stage']
    predicted_cancer = comparison.loc[comparison['results']==1]
    predicted_cancer = predicted_cancer.loc[comparison['diagnosis']==1]
    counts = {'I':0,'IA':0,'IB':0,'II':0,'IIA':0,'IIB':0,'III':0,'IV':0}
    for each in counts:
        diagnosed = (predicted_cancer['type'] == each).sum()
        total = (testpatients['stage'] == each).sum()
        counts[each] = diagnosed/total * 100
    plt.figure(figsize=(15,5))
    plt.grid()
    plt.bar(counts.keys(), counts.values(), width=.5, color='b')
def main():
    dataset = importData()
    X_raw,y = formatData(dataset)
    X = normalise(X_raw)
    expected,results = svm(X,y)
    pca(X,y,dataset)
    t_sne(X,y,dataset)
    typeAnalysis(expected, results,dataset)
main()
        