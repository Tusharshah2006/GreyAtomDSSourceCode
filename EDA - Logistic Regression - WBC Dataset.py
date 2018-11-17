
# coding: utf-8

# In[108]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
import operator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, RFE, RFECV
from sklearn.model_selection import train_test_split as tts, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[55]:


os.chdir("D:/GreyAtom/Datasets")

df = pd.read_csv("wbc.csv")


# In[56]:


le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])


# In[57]:


df = df.drop(["Unnamed: 32"], axis = 1)


# In[58]:


df = df.drop(["id"], axis = 1)
df.head()


# In[59]:


df["diagnosis"].value_counts()


# In[60]:


scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# In[61]:


corr = df.corr()
corr["diagnosis"]


# In[62]:


sns.set_context("notebook", font_scale = 1.0, rc = {"lines.linewidth":4.5})
plot.figure(figsize = (31, 15))
a = sns.heatmap(corr, annot = True, fmt = ".2f")


# In[66]:


X = df.drop(["diagnosis"], axis = 1)
y = df["diagnosis"]


# In[67]:


test = SelectKBest(score_func=chi2, k=8)
fit = test.fit(X, y)

features = list(X.columns)
scores = list(fit.scores_)

dictionary = dict(zip(features, scores))
sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)


# In[116]:


model = LogisticRegression()
rfe = RFE(model, 8)
fit = rfe.fit(X, y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))


# In[117]:


features = list(X.columns)
scores = list(fit.support_)
import operator
d = {}
d = dict(zip(features, scores))
sorted(d.items(), key=operator.itemgetter(1))


# In[118]:


feature_list = []
for x in d:
    if d[x] == True:
        feature_list.append(x)


# In[119]:


X_model = df[feature_list]
y_model = df["diagnosis"]

logreg = LogisticRegression()
cv_log_ref = cross_val_score(logreg, X_model, y_model, cv = 5)
cv_log_ref.min()

