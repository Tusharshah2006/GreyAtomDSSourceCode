
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split as tts, cross_val_score


# In[49]:


import os
os.chdir("D:/GreyAtom/Datasets/09292018")

df = pd.read_csv("auto.csv")
df.head()


# In[41]:


df_orogin = pd.get_dummies(df)
df_orogin


# In[42]:


df_orogin = df_orogin.drop(["origin_Asia"], axis=1)


# In[43]:


df_orogin


# In[44]:


X = df_orogin.drop(["mpg"], axis = 1)
y = df_orogin["mpg"]
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)


# In[45]:


reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
reg.score(X_test, y_test)


# In[46]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse


# In[47]:


df_orogin = df_orogin.drop(["origin_Europe", "origin_US"], axis=1)
df_orogin


# In[48]:


X = df_orogin.drop(["mpg"], axis = 1)
y = df_orogin["mpg"]
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
reg_score = reg.score(X_test, y_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(reg_score, rmse)


# In[50]:


df_all_columns = pd.get_dummies(df)
df_all_columns


# In[56]:


X = df_all_columns.drop(["mpg"], axis = 1)
y = df_all_columns["mpg"]
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)

reg = Ridge(alpha=0.1, normalize=True)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
reg_score = reg.score(X_test, y_test)
print(reg_score)


# In[61]:


new = []
l = np.linspace(0, 2, 11)

for z in l:
    ridge = Ridge(alpha = z, normalize=True)
    ridge.fit(X_train, y_train)
    new.append(ridge.score(X_test, y_test))
    
print(new)

