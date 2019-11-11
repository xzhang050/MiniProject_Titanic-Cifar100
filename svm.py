import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from feature_engineering import data_processing




combined_train_test = data_processing()

train_data = combined_train_test[:891]
test_data = combined_train_test[891:]

titanic_train_data_X = train_data.drop(['Survived'],axis=1)[:650]
titanic_train_data_Y = train_data['Survived'][:650]
titanic_test_data_X = train_data.drop(['Survived'],axis=1)[650:]
titanic_test_data_Y = train_data['Survived'][650:]
# titanic_test_data_X = test_data.drop(['Survived'],axis=1)

svm = SVC(kernel='linear', C=0.025)
svm = svm.fit(titanic_train_data_X, titanic_train_data_Y)
predictions = svm.predict(titanic_test_data_X)
scores = svm.score(titanic_test_data_X,titanic_test_data_Y)
print("test accuracy: {}".format(scores))
#score 0.8340248962655602


# clf = DecisionTreeClassifier()
# clf = clf.fit(titanic_train_data_X, titanic_train_data_Y)
# predictions = clf.predict(titanic_test_data_X)
# scores = clf.score(titanic_test_data_X,titanic_test_data_Y)
# print("test accuracy: {}".format(scores))
#0.759
# StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
# StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')
