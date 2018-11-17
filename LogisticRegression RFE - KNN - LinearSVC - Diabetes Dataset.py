
# coding: utf-8

# In[78]:


import os
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, classification_report, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split as tts, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.svm import LinearSVC

os.chdir("D:/GreyAtom/Datasets/09292018")

df = pd.read_csv("diabetes.csv")
df.head()


# In[79]:


df["insulin"].replace (0, df["insulin"].mean(), inplace=True)


# In[80]:


df["triceps"].replace (0, df["triceps"].mean(), inplace=True)


# In[81]:


df.head()


# In[82]:


df["bmi"].replace (0, df["bmi"].mean(), inplace=True)


# In[83]:


df.head()


# In[84]:


X = df.drop(["diabetes"], axis = 1)
y = df["diabetes"]


# In[85]:


df.corr()


# In[86]:


logreg = LogisticRegression()
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)
logreg.fit(X_train, y_train)


# In[87]:


logreg.score(X_test, y_test)


# In[88]:


X1 = df[["glucose", "bmi", "age"]]
y1 = df["diabetes"]


# In[89]:


logreg1 = LogisticRegression()
X1_train, X1_test, y1_train, y1_test = tts(X1, y1, test_size = 0.3, random_state = 42)
logreg1.fit(X1_train, y1_train)
logreg1.score(X1_test, y1_test)
#y1_pred_proba = logreg1.predict_proba(X1_test)[:, 1]
#auc(y1_test, y1_pred_proba)


# In[90]:


X2 = df[["pregnancies", "glucose", "insulin", "age"]]
y2 = df["diabetes"]

logreg2 = LogisticRegression()
X2_train, X2_test, y2_train, y2_test = tts(X2, y2, test_size = 0.3, random_state = 42)
logreg2.fit(X2_train, y2_train)
logreg2.score(X2_test, y2_test)


# In[91]:


X3 = df[["pregnancies", "glucose", "insulin", "age"]]
y3 = df["diabetes"]

knn = KNeighborsClassifier()
cv_results = cross_val_score(knn, X3, y3, cv = 5)
cv_results


# In[92]:


logreg3 = LogisticRegression()
cv_log_ref = cross_val_score(logreg3, X3, y3, cv = 5)
cv_log_ref.min()


# In[93]:


svc = LinearSVC()
cv_svc = cross_val_score(svc, X3, y3, cv = 5)
cv_svc

