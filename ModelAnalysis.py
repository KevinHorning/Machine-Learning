import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import hamming_loss
from sklearn.multiclass import OneVsRestClassifier
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

trainData1 = pd.read_csv('TrainData1.txt', sep="\t", header=None)
trainData2 = pd.read_csv('TrainData2.txt', sep="  ", header=None)
trainData3 = pd.read_csv('TrainData3.txt', sep="\t", header=None)
trainData4 = pd.read_csv('TrainData4.txt', sep="  ", header=None)
trainData5 = pd.read_csv('TrainData5.txt', sep="  ", header=None)
trainData6 = pd.read_csv('TrainData6.txt', sep="  ", header=None)

trainData1 = trainData1.replace(1.0000000000000001e+99, np.nan, regex=True)
trainData2 = trainData2.replace(1.0000000000000001e+99, np.nan, regex=True)
trainData3 = trainData3.replace(1.0000000000000001e+99, np.nan, regex=True)
trainData4 = trainData4.replace(1.0000000000000001e+99, np.nan, regex=True)
trainData5 = trainData5.replace(1.0000000000000001e+99, np.nan, regex=True)
trainData6 = trainData6.replace(1.0000000000000001e+99, np.nan, regex=True)

from fancyimpute import KNN
trainData1 = KNN(k=3).fit_transform(trainData1)
trainData3 = KNN(k=3).fit_transform(trainData3)

trainLabels1 = pd.read_csv('TrainLabel1.txt', sep="\t", header=None)
trainLabels2 = pd.read_csv('TrainLabel2.txt', sep="\t", header=None)
trainLabels3 = pd.read_csv('TrainLabel3.txt', sep="\t", header=None)
trainLabels4 = pd.read_csv('TrainLabel4.txt', sep="\t", header=None)
trainLabels5 = pd.read_csv('TrainLabel5.txt', sep="\t", header=None)
trainLabels6 = pd.read_csv('TrainLabel6.txt', sep="\t", header=None)

def runSuite(data, labels, multi):
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        x = data
        y = labels

        x = StandardScaler().fit_transform(x)

        c = findBestC(x, y, multi)
        print('Logistic Regression where C =', c)
        if multi:
            model = OneVsRestClassifier(LogisticRegression(solver = 'lbfgs', C = c))
        else:
            model = LogisticRegression(C = c)
        runReports(model, x, y)

        n = findBestN(x, y, multi)
        print('Random Forest Classifier with', n,'estimators')
        if multi:
            model = OneVsRestClassifier(RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0))
        else:
            model = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)
        runReports(model, x, y)

        kern = findBestKernel(x, y, multi)
        print('Support Vector Machine with kernel =',kern)
        if multi:
            model = OneVsRestClassifier(SVC(gamma='auto', probability=True, kernel = kern))
        else:
            model = SVC(gamma='auto', probability=True, kernel = kern)
        runReports(model, x, y)

        k = findBestK(x, y, multi)
        print('KNN Classifier where k=', k)
        if multi:
            model = OneVsRestClassifier(SVC(gamma='auto', probability=True, kernel = kern))
        else:
            model = KNeighborsClassifier(n_neighbors=k)
        runReports(model, x, y)

def runReports(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    model.fit(x_test, y_test) 

    y_pred = model.predict(x_test)
    print('Model Score: ',model.score(x_test, y_test))
    
    print('Classification Report: \n', classification_report(y_test, y_pred))
    
    temp = np.array(y_test[0])
    w = 0
    mode = y_test.mode()
    while w < temp.size:
        if temp[w] == mode.iloc[0, 0]:
            temp[w] = True
        else:
            temp[w] = False
        w+=1
    temp = temp.astype('bool')
    
    print('ROC Curve: ')
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(temp, y_pred_proba)
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr, tpr, label='knn')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.show()
    
    print('Cross Validation Scores: ')
    kFolds = np.array([3, 5])
    for i in range (0, kFolds.size):
        scores = cross_val_score(model, x_test, y_test, cv=kFolds[i])
        print(kFolds[i], 'folds: ', scores)
        
    print("Hamming Loss: ", hamming_loss(y_test, y_pred))
    print()

def findBestC(x, y, multi):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    maxScore = 0
    c = 0
    for i in range(1, 11):
        if multi:
            model = OneVsRestClassifier(LogisticRegression(C=(i/10)))
        else:
            model = LogisticRegression(C=(i/10))
        model.fit(x_test, y_test) 
        
        curr = model.score(x_test, y_test)
        if (curr > maxScore):
            maxScore = curr
            c = i
    return c

def findBestN(x, y, multi):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    maxScore = 0
    n = 0
    for i in range (1, 10):
        if multi:
            model = OneVsRestClassifier(RandomForestClassifier(n_estimators=i*10, max_depth=2, random_state=0))
        else:
            model = RandomForestClassifier(n_estimators=i*10, max_depth=2, random_state=0)
        model.fit(x_test, y_test) 
        
        curr = model.score(x_test, y_test)
        if (curr > maxScore):
            maxScore = curr
            n = i
    return n

def findBestKernel(x, y, multi):
    kernels = ['rbf', 'poly', 'sigmoid']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    maxScore = 0
    index = 0
    for i in kernels:
        if multi:
            model = OneVsRestClassifier(SVC(gamma='auto', probability=True, kernel = i))
        else:
            model = SVC(gamma='auto', probability=True, kernel = i)
        model.fit(x_test, y_test) 
        
        curr = model.score(x_test, y_test)
        if (curr > maxScore):
            maxScore = curr
            index = i
    return index

def findBestK(x, y, multi):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    maxScore = 0
    k = 0
    for i in range(2, 10):
        if multi:
            model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=i))
        else:
            model = KNeighborsClassifier(n_neighbors=i)
        model.fit(x_test, y_test) 
        
        curr = model.score(x_test, y_test)
        if (curr > maxScore):
            maxScore = curr
            k = i
    return k

def runPCA(x, y, multi):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, y], axis = 1)
    print('Dimensional Reduction: ')
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_title('2 component PCA', fontsize = 20)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    
    if (y[0].nunique() < 8 and not multi):
        targets = [1, 2, 3, 4, 5, 6, 7]
        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf[0] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c = color
                       , s = 50)
        ax.legend(targets)
        ax.grid()
    else:
        ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'])

def runMissing(data):
    print("KNN Imputation\n")
    print('********************************************************')
    k = 0
    score = 0
    for i in range(1, 10):
        print("K =", i)
        nData = pd.DataFrame(KNN(k=i).fit_transform(data))

        x = nData.drop([nData.columns.size - 1], axis = 1)
        y = nData[nData.columns.size - 1]
        
        model = LinearRegression()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        model.fit(x_test, y_test)
        
        curr = model.score(x_test, y_test)
        if (score < curr):
            score = curr
            k = i

        print('Linear Regression Model Score:',curr)
        print('\n********************************************************')
    print('Best k =', k, 'with score of', score)

runSuite(trainData1, trainLabels1, False)
runSuite(trainData2, trainLabels2, False)
runSuite(trainData3, trainLabels3, False)
runSuite(trainData4, trainLabels4, False)
runSuite(trainData5, trainLabels5, False)
runSuite(trainData6, trainLabels6, False)

missingData1 = pd.read_csv('MissingData1.txt', sep="\t", header=None)
missingData2 = pd.read_csv('MissingData2.txt', sep="\t", header=None)
missingData3 = pd.read_csv('MissingData3.txt', sep="\t", header=None)

missingData1 = missingData1.replace(1.0000000000000001e+99, np.nan, regex=True)
missingData2 = missingData2.replace(1.0000000000000001e+99, np.nan, regex=True)
missingData3 = missingData3.replace(1.0000000000000001e+99, np.nan, regex=True)

runMissing(missingData1)
runMissing(missingData2)
runMissing(missingData3)

mlTestData = pd.read_csv('MultLabelTestData.txt', sep="\t", header=None)
mlTrainData = pd.read_csv('MultLabelTrainData.txt', sep="\t", header=None)
mlTrainLabel = pd.read_csv('MultLabelTrainLabel.txt', sep="\t", header=None)

runSuite(mlTrainData, mlTrainLabel, True)


