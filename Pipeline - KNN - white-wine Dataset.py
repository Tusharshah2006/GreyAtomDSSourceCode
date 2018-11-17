
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split as tts, cross_val_score
from sklearn.preprocessing import Imputer, scale, StandardScaler
from sklearn.pipeline import Pipeline

import os
os.chdir("D:/GreyAtom/Datasets/09292018")

df = pd.read_csv("white-wine.csv")
df.head()


# In[12]:


X = df.drop(["quality"], axis = 1)
y = df["quality"]


# In[13]:


X_scaled = scale(X)
X_scaled[1]


# In[14]:


steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=6))]


# In[15]:


pipeline = Pipeline(steps)


# In[19]:


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)
knn_scaled = pipeline.fit(X_train, y_train)

y_pred = knn_scaled.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[20]:


accuracy_score(y_test, y_pred)

