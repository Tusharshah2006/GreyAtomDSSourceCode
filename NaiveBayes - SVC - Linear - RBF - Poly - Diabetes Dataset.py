
# coding: utf-8

# In[11]:


import os
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plot

from sklearn.metrics import auc, classification_report, confusion_matrix, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split as tts, cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Imputer
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC


# In[12]:


os.chdir("D:/GreyAtom/Datasets/09292018")
df = pd.read_csv("diabetes.csv")


# In[13]:


X = df.drop(["diabetes"], axis=1)
y = df["diabetes"]


# In[18]:


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42, stratify=y)


# In[23]:


nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(classification_report(y_test, y_pred))


# In[22]:


svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred))


# In[24]:


svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:


svc = SVC(kernel='poly')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred))

