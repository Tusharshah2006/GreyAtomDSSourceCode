
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


# In[3]:


df = pd.read_csv("diabetes.csv")


# In[4]:


df.head()


# In[5]:


list(df)


# In[6]:


X = df.drop(["diabetes"], axis = 1)
y = df["diabetes"]


# In[7]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[9]:


test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)


# In[10]:


import numpy as np


# In[26]:


features = list(X.columns)
scores = list(fit.scores_)
import operator
d = {}
d = dict(zip(features, scores))
sorted(d.items(), key=operator.itemgetter(1))


# In[12]:


features = fit.transform(X)


# In[13]:


print (features)


# In[27]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[29]:


model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))


# In[31]:


features = list(X.columns)
scores = list(fit.support_)
import operator
d = {}
d = dict(zip(features, scores))
sorted(d.items(), key=operator.itemgetter(1))


# In[32]:


from sklearn.linear_model import Ridge


# In[33]:


ridge = Ridge(alpha=1.0)
ridge.fit(X,y)


# In[34]:


def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)


# In[35]:


print ("Ridge model:", pretty_print_coefs(ridge.coef_))

