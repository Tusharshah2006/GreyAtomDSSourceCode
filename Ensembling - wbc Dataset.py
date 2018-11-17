
# coding: utf-8

# In[36]:


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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import VotingClassifier


# In[20]:


os.chdir("D:/GreyAtom/Datasets")
df = pd.read_csv("wbc.csv")


# In[21]:


le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])
df = df.drop(["Unnamed: 32", "id"], axis = 1)


# In[22]:


scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# In[23]:


X = df[['concavity_mean', 'concave points_mean', 'fractal_dimension_mean', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'concave points_worst']]
y = df["diagnosis"]


# In[24]:


#Create a dictionary of possible parameters
params = {"n_neighbors" : np.arange(1, 50)}

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, params, cv = 5)
knn_cv.fit(X, y)


# In[25]:


print(knn_cv.best_params_)
print(knn_cv.best_score_)
print(knn_cv.best_estimator_)


# In[32]:


params = {"C" : np.arange(1, 50)}
logreg_grid = LogisticRegression()
logreg_cv = GridSearchCV(logreg_grid, params, cv=5)
logreg_cv.fit(X, y)

print(logreg_cv.best_params_)
print(logreg_cv.best_score_)
print(logreg_cv.best_estimator_)


# In[31]:


#Create a dictionary of possible parameters
params_grids = {'C': [0.01, 0.1, 0.001, 1, 10, 100],
          'gamma': [0.0001, 0.001, 0.01, 0.1],
          'kernel':['linear','rbf'] }

#Create the GridSearchCV object
grid_clf = GridSearchCV(SVC(class_weight='balanced'), params_grids)

#Fit the data with the best possible parameters
grid_clf = grid_clf.fit(X, y)

#Print the best estimator with it's parameters
print (grid_clf.best_estimator_)
print(grid_clf.best_score_)
print(grid_clf.best_params_)


# In[40]:


classifiers = []
classlogregtup = ("LogisticRegression", logreg_cv)
classknntup = ("KNeighborsClassifier", knn_cv)
classsvctup = ("SVC", grid_clf)

classifiers.append(classknntup)
classifiers.append(classlogregtup)
classifiers.append(classsvctup)


# In[41]:


classifiers


# In[43]:


vc = VotingClassifier(estimators=classifiers)
vc.fit(X, y)
vc.estimators_

