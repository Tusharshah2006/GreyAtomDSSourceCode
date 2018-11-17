
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split as tts, cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline


# In[2]:


import os
os.chdir("D:/GreyAtom/Datasets/09292018")

df = pd.read_csv("diabetes.csv")
df.head()


# In[3]:


df["insulin"].replace(0, np.nan, inplace=True)
df["triceps"].replace(0, np.nan, inplace=True)
df["bmi"].replace(0, np.nan, inplace=True)


# In[4]:


df.shape


# In[13]:


X = df.drop(["diabetes"], axis = 1)
y = df["diabetes"]

imp = Imputer(missing_values="NaN", strategy="mean")
logreg = LogisticRegression()

steps = [("imputation", imp), ("logistic_reg", logreg)]
pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)


# In[14]:


pipeline.fit(X_train, y_train)


# In[15]:


y_pred = pipeline.predict(X_test)
pipeline.score(X_test, y_test)

