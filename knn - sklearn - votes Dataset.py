
# coding: utf-8

# In[57]:


import os
os.chdir("D:/GreyAtom/Datasets/09292018")


# In[58]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts, cross_val_score


# In[59]:


vots = pd.read_csv("votes.csv")
votes.info()
votes.head()


# In[60]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[61]:


X = votes.drop(["party"], axis = 1)
y = votes["party"]


# In[62]:


X.shape, y.shape


# In[63]:


votes = votes.apply(LabelEncoder().fit_transform)

X = votes.drop(["party"], axis = 1)
y = votes["party"]


# In[64]:


knn.fit(X, y)


# In[78]:


votes_train = votes.iloc[:400]
votes_test = votes.iloc[400:]


# In[79]:


X = votes_train.drop(["party"], axis = 1)
y = votes_train["party"]


# In[80]:


knn_model = knn.fit(X, y)


# In[81]:


y_test = votes_test["party"]
votes_test_pred = votes_test.drop(["party"], axis = 1)


# In[82]:


y_pred = knn_model.predict(votes_test_pred)


# In[83]:


y_pred = list(y_pred)
y_test = list(y_test)


# In[84]:


mis_classified = []
for x_mis, y_mis in zip(y_test, y_pred):
    if x_mis!=y_mis:
        mis_classified.append(x_mis)
print(1-(len(mis_classified)/len(y_test)))


# In[86]:


knn_cv_errors = []

for k in np.arange(1, 9):
    knn = KNeighborsClassifier(n_neighbors=k)
    cvscores = cross_val_score(knn, X, y, cv = 3, scoring = 'accuracy')
    knn_cv_errors.append(cvscores.mean())
print(knn_cv_errors)

