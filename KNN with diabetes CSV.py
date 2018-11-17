
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split as tts, cross_val_score
from sklearn.metrics import mean_squared_error


# In[2]:


import os
os.chdir("D:/GreyAtom/Datasets/09292018")

gm2008 = pd.read_csv("gm_2008_region.csv")
gm2008.head()


# In[4]:


gm2008_region = pd.get_dummies(gm2008)
print(gm2008_region.columns)


# In[7]:


df = pd.read_csv("diabetes.csv")
df.head()
df.info()

X = df.drop(["diabetes"], axis = 1)
y = df["diabetes"]


# In[15]:


knn_cv_errors = []

for k in np.arange(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    cvscores = cross_val_score(knn, X, y, cv = 5, scoring = 'accuracy')
    knn_cv_errors.append(cvscores.mean())

knn_cv_errors

