
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split as tts, cross_val_score


# In[12]:


import os
os.chdir("D:/GreyAtom/Datasets/09292018")

df = pd.read_csv("diabetes.csv")
df.head()


# In[13]:


X = df.drop(["diabetes"], axis = 1)
y = df["diabetes"]


# In[14]:


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)


# In[15]:


knn = KNeighborsClassifier(n_neighbors=6)
knn_model = knn.fit(X_train, y_train)


# In[16]:


y_pred = knn_model.predict(X_test)


# In[19]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[42]:


y_pred_proba = knn.predict_proba(X_test)[:,1]
print(y_pred_proba[3], y_pred[3])

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc_score(y_test, y_pred_proba)


# In[20]:


Logistic = LogisticRegression()
Logistic.fit(X_train, y_train)


# In[21]:


ylog_pred = Logistic.predict(X_test)


# In[24]:


ylog_pred[:10]


# In[25]:


list(y_test[:10])


# In[38]:


print(confusion_matrix(y_test, ylog_pred))
print(classification_report(y_test, ylog_pred))

ylog_pred_proba = Logistic.predict_proba(X_test)[:,1]
print(ylog_pred_proba[3], ylog_pred[3])


# In[39]:


fpr, tpr, thresholds = roc_curve(y_test, ylog_pred_proba)
roc_auc_score(y_test, ylog_pred_proba)

