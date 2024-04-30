# libraries for SVC
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# libraries for crossfold
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Read MNIST data
import pandas as pd
import numpy as np

data = pd.read_csv('heart_2022_no_nans_preprocessed.csv')

# make two variables - X and y
y = data.iloc[:, 9]
X = data.drop('HadHeartAttack', axis=1)
X = X.drop('State', axis=1)
n, p = X.shape # number of samples and features

kf = KFold(n_splits=5, random_state=None, shuffle=False) # split data into 5 folds

print("Calculating accuracy for SVM with RBF kernel\n")
accuracy = 0

# train 4 folds, test 1 fold experiment 5 times
for k, (train_index, test_index) in enumerate(kf.split(X,y)):
    # get train fold
    X_train = X.iloc[train_index, :] 
    y_train = y.iloc[train_index]

    # get test fold
    X_test = X.iloc[test_index,:] 
    y_test = y.iloc[test_index]

    # apply SVM with rbf kernel
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto', class_weight='balanced'))
    clf.fit(X_train, y_train)
    results = clf.predict(X_test)

    # compare predictions and truth
    count = 0
    correctlyGuessed = 0
    for i in results:
        if(i == y_test.iloc[count]):
            correctlyGuessed += 1
        count+=1
    accuracy += correctlyGuessed/len(y_test)


    # print results
    print("\tFold %d" % (k+1))
    print("\tCorrectly classified: %d" % (correctlyGuessed))
    print("\tIncorrectly classified: %d" % (len(y_test) - correctlyGuessed))
    print("\tAccuracy: %f" % (correctlyGuessed/len(y_test)))
    print("\tAccuracy: %f" % (correctlyGuessed/len(y_test)))
    print(classification_report(y_test,results))

accuracy /= 5
print(" Average Accuracy: %f" % (accuracy))