
# coding: utf-8

# In[81]:


import os
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, classification_report, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split as tts, cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.svm import LinearSVC


# In[87]:


os.chdir("D:/GreyAtom/Datasets/09292018")

df = pd.read_csv("gm_2008_region.csv")
df.head()

df = pd.get_dummies(df)

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# In[88]:


X = df.drop(["life"], axis = 1)
y = df["life"]


# In[89]:


def get_frwd_pass_features(cols):
    columnList = []
    rmseList = []
    for col in cols:
        columnList.append(col)
        X_train, X_test, y_train, y_test = tts(df[columnList], y, test_size = 0.3, random_state = 42)
        LinReg = LinearRegression()
        LinReg.fit(X_train, y_train)
        
        y_pred = LinReg.predict(X_test)
        rsquare = LinReg.score(X_test, y_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmseList.append(rmse)
    print(rmseList)


# In[85]:


get_frwd_pass_features(X.columns)

X = df_scaled.drop(["life"], axis = 1)
y = df_scaled["life"]

get_frwd_pass_features(X.columns)

