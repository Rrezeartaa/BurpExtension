from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib 
import pickle

data= pd.read_csv("./dataset/kaggleDataset/sqliv2/sqliv2.csv",encoding='utf-16')
data.info()
data.head()

vectorizer = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english')) 
vectorizer.fit(data['Sentence'].values.astype('U')) 
print(vectorizer.vocabulary_) 
vector= vectorizer.transform(data['Sentence'].values.astype('U'))
posts= vector.toarray()
print(vector.shape)
print(type(vector))
print(posts) 
transformed_posts=pd.DataFrame(posts)
df=pd.concat([data,transformed_posts],axis=1)
X=df[df.columns[2:]]
y=df['Label']
df.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X.head()

X_test.info()


regressor = GradientBoostingClassifier()
regressor.fit(X_train, y_train)

y_pred_log_reg=regressor.predict(X_test)
accuracy_score(y_test, y_pred_log_reg)
print(y_pred_log_reg)


accuracy=accuracy_score(y_test, y_pred_log_reg)
precision=precision_score(y_test, y_pred_log_reg, zero_division=1)
recall=recall_score(y_test, y_pred_log_reg, zero_division=1)
print(" Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(accuracy, precision, recall))





