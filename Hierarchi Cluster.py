
# coding: utf-8

# In[21]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage

X = np.array([[5,3],  
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],])


# In[13]:


plt.scatter(X[:,0], X[:,1])


# In[14]:


linked = linkage(X, 'single')
labelList = range(1, 11)
plt.figure(figsize=(10, 7))  
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show() 


# In[17]:


os.chdir("D:/GreyAtom/Datasets")
df = pd.read_csv("shopping_data.csv")
df_copy = df


# In[25]:


df = df.drop(['CustomerID', 'Genre', 'Age'], 1)


# In[26]:


plt.figure(figsize=(10, 7))
plt.title("Customer Dendrogram")

dend = sch.dendrogram(sch.linkage(df, method='ward'))

