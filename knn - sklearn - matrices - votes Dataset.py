
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# In[5]:


import os
os.chdir("D:/GreyAtom/Datasets/09292018")

votes = pd.read_csv("votes.csv")


# In[6]:


votes = votes.apply(LabelEncoder().fit_transform)


# In[7]:


X = votes.drop(["party"], axis = 1)
y = votes["party"]


# In[8]:


X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)


# In[9]:


knn = KNeighborsClassifier(n_neighbors=6)
knn_model = knn.fit(X_train, y_train)


# In[14]:


y_pred = knn_model.predict(X_test)


# In[16]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[13]:


neighbours = np.arange(1, 9)
print(neighbours)
train_accuracy = np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))

for i, k in enumerate(neighbours):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

