
# coding: utf-8

# In[15]:


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

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.svm import SVC
from sklearn.cluster import KMeans


# In[4]:


os.chdir("D:/GreyAtom/Datasets/09292018")
df = pd.read_csv("diabetes.csv")
df_copy = df


# In[5]:


df = df.drop(['diabetes'],1)


# In[17]:


kmeans = KMeans(n_clusters=2)
kmeans.fit(df)


# In[8]:


y = df_copy['diabetes']
y = np.array(y)


# In[13]:


mis_classifies = np.where(kmeans.labels_ != y)
len(mis_classifies[0])


# In[16]:


plt.figure(figsize=(10, 7))
plt.title("Customer Dendrogram")

dend = sch.dendrogram(sch.linkage(df, method='ward'))

