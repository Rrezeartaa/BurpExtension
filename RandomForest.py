import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from nltk.corpus import stopwords 
from time import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib 
import pickle

print(__doc__)
bag_clf = RandomForestClassifier()
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]

grid_search = GridSearchCV(bag_clf, param_grid=param_grid, cv=5, scoring= 'recall')
start = time()
grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

grid_search.cv_results_

grid_search.best_params_

grid_search.best_estimator_

bag_clf = RandomForestClassifier(n_estimators=10, max_features= 4, bootstrap= False)
bag_clf.fit(X_train, y_train)
y_pred_random = bag_clf.predict(X_test)

accuracy=accuracy_score(y_test, y_pred_random)
precision=precision_score(y_test, y_pred_random, zero_division=1)
recall=recall_score(y_test, y_pred_random, zero_division=1)
print(" Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(accuracy, precision, recall))