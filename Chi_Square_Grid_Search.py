
# coding: utf-8

# In[42]:


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


# In[43]:


os.chdir("D:/GreyAtom/Datasets")

df = pd.read_csv("output.csv")
df.head()


# In[44]:


le = LabelEncoder()
df = df.apply(le.fit_transform)


# In[45]:


df.head()


# In[46]:


X = df.drop(["party"], axis = 1)
y = df["party"]


# In[47]:


test = SelectKBest(score_func=chi2, k=8)
fit = test.fit(X, y)


# In[48]:


features = list(X.columns)
scores = list(fit.scores_)


# In[49]:


dictionary = dict(zip(features, scores))
sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)


# In[64]:


chi2_features = ["physician", "salvador", "aid", "budget", "education", "crime", "duty_free_exports", "missile"]

X1 = df[chi2_features]
y1 = df["party"]


# In[58]:


params = {"n_neighbors" : np.arange(1, 50)}


# In[65]:


knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, params, cv = 5)
knn_cv.fit(X1, y1)


# In[66]:


knn_cv.best_params_


# In[67]:


knn_cv.best_score_


# In[68]:


logreg = LogisticRegression()
cv_log_reg = cross_val_score(logreg, X1, y1, cv=5)
cv_log_reg.min()


# In[74]:


params = {"C" : np.arange(1, 50)}
logreg_grid = LogisticRegression()
logreg_cv = GridSearchCV(logreg_grid, params, cv=5)
logreg_cv.fit(X1, y1)

print(logreg_cv.best_params_)
print(logreg_cv.best_score_)

