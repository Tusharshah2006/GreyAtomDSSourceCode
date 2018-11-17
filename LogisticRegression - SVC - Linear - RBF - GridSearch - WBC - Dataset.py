
# coding: utf-8

# In[58]:


import os
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, classification_report, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split as tts, cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.svm import LinearSVC


# In[59]:


os.chdir("D:/GreyAtom/Datasets")
df = pd.read_csv("wbc.csv")


# In[60]:


le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])
df = df.drop(["Unnamed: 32", "id"], axis = 1)


# In[61]:


scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# In[62]:


X = df[['concavity_mean', 'concave points_mean', 'fractal_dimension_mean', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'concave points_worst']]
y = df["diagnosis"]


# In[63]:


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)


# In[64]:


Logistic = LogisticRegression()
Logistic.fit(X_train, y_train)
Logistic.score(X_test, y_test)


# In[65]:


svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
svc.score(X_test, y_test)


# In[66]:


rbf = SVC(kernel='rbf')
rbf.fit(X_train, y_train)
rbf.score(X_test, y_test)


# In[74]:


#Create a dictionary of possible parameters
params_grids = {'C': [0.01, 0.1, 0.001, 1, 10, 100],
          'gamma': [0.0001, 0.001, 0.01, 0.1],
          'kernel':['linear','rbf'] }

#Create the GridSearchCV object
grid_clf = GridSearchCV(SVC(class_weight='balanced'), params_grids)

#Fit the data with the best possible parameters
grid_clf = grid_clf.fit(X_train, y_train)

#Print the best estimator with it's parameters
print (grid_clf.best_estimator_)


# In[72]:


print(grid_clf.best_score_)

