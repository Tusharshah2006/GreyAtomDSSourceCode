
# coding: utf-8

# In[14]:


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
from sklearn.tree import DecisionTreeClassifier, tree

from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.svm import LinearSVC


# In[15]:


os.chdir("D:/GreyAtom/Datasets")
df = pd.read_csv("wbc.csv")


# In[16]:


le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])

df = df.drop(["Unnamed: 32", "id"], axis = 1)

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# In[17]:


#X = df[['concavity_mean', 'concave points_mean', 'fractal_dimension_mean', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'concave points_worst']]
X = df[['radius_mean', 'concave points_mean']]
y = df["diagnosis"]


# In[18]:


dt = DecisionTreeClassifier(max_depth=6, criterion="gini", random_state=42)


# In[19]:


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42, stratify=y)


# In[20]:


dt.fit(X_train, y_train)


# In[21]:


y_pred = dt.predict(X_test)


# In[24]:


score = dt.score(X_test, y_test)
acc = accuracy_score(y_test, y_pred)
print(score, acc)


# In[25]:


print(classification_report(y_test, y_pred))


# In[35]:


#Create a dictionary of possible parameters
params_grids = {'max_depth': np.arange(1, len(X))}

#Create the GridSearchCV object
grid_clf = GridSearchCV(DecisionTreeClassifier(), params_grids)

#Fit the data with the best possible parameters
grid_clf = grid_clf.fit(X_train, y_train)

#Print the best estimator with it's parameters
print (grid_clf.best_estimator_)
print(grid_clf.best_score_)

