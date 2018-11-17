
# coding: utf-8

# In[20]:


import os
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, classification_report, confusion_matrix, mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split as tts, cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import VotingClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, tree, DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier


# In[21]:


os.chdir("D:/GreyAtom/Datasets")
df = pd.read_csv("wbc.csv")


# In[22]:


X = df.drop(["diagnosis", "Unnamed: 32"], axis = 1)
y = df["diagnosis"]


# In[23]:


X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, stratify=y, random_state=1)


# In[24]:


dtc = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=1)


# In[25]:


bc = BaggingClassifier(base_estimator=dtc, n_estimators=300, n_jobs=1)
bc.fit(X_train, y_train)


# In[26]:


y_pred = bc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# In[27]:


dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# In[31]:


bc = BaggingClassifier(base_estimator=dtc, n_estimators=300, oob_score=True, n_jobs=-1)

bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
bc.oob_score_

