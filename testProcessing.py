# runs KNN and Logistic Regression on a multilabel dataset and calculates hamming losses 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import hamming_loss 
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('MultLabelTrainData.txt', sep="\t", header=None)
labels = pd.read_csv('MultLabelTrainLabel.txt', sep="\t", header=None)
pd.DataFrame(data)
pd.DataFrame(labels)

def KNearestNeighbors():
    Ks = [1, 3, 5, 15]
  
    losses = []
    kf = KFold(n_splits = 10)
    for k in Ks:    
        foldLosses = []
        for train_index, test_index in kf.split(data):
            x_train, x_test, y_train, y_test = [data.iloc[i] for i in train_index], [data.iloc[i] for i in test_index], [labels.iloc[i] for i in train_index], [labels.iloc[i] for i in test_index]
            
            KNN = KNeighborsClassifier(n_neighbors = k)
            KNN.fit(x_train, y_train)
            y_pred = KNN.predict(x_test)
            foldLosses.append(hamming_loss(np.asarray(y_test), y_pred))
            
        losses.append(sum(foldLosses)/len(foldLosses))
    print(losses)

def LogRegression(): 
    logreg = LogisticRegression()
    kfolds = [2, 4, 6, 10] 

    kf = KFold(n_splits = 3)
    losses = [] 
    for train_index, test_index in kf.split(data):
        x_train, x_test, y_train, y_test = [data.iloc[i] for i in train_index],  [data.iloc[i] for i in test_index], [labels.iloc[i] for i in train_index], np.asarray([labels.iloc[i] for i in test_index])
        y_pred = []

        count = 0
        for col in range(len(y_test[0]) - 1):
            y_trainDF = pd.DataFrame(y_train)
            logreg.fit(x_train, np.ravel(y_trainDF.iloc[:,col:col+1]))
            y_predCol = logreg.predict(x_test)
            count += 1
            y_pred.append(y_predCol)
 
        lastCol = []
        rowSize = len(y_pred[0])
        [lastCol.append(0) for i in range(rowSize)]        
        y_pred.append(lastCol)
        
        losses.append(hamming_loss(np.asarray(np.transpose(y_test)),np.asarray(y_pred)))
        
    print(losses)
        
KNearestNeighbors()
LogRegression()