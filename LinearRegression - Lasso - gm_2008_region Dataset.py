
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split as tts, cross_val_score
from sklearn.metrics import mean_squared_error


# In[12]:


import os
os.chdir("D:/GreyAtom/Datasets/09292018")

gm2008 = pd.read_csv("gm_2008_region.csv")
gm2008.head()


# In[13]:


X = gm2008["fertility"]
y = gm2008["life"]

plot.scatter(X, y)


# In[14]:


linreg = LinearRegression()

y = y.values.reshape(-1, 1)
X = X.values.reshape(-1, 1)


# In[15]:


linreg.fit(X, y)


# In[17]:


y_pred = linreg.predict(X)
y_pred


# In[18]:


gm2008["life"]


# In[19]:


linreg.score(X, y)


# In[24]:


plot.scatter(X, y)
plot.plot(X, y_pred, color='Black', linewidth=3)
plot.show()


# In[27]:


gm2008 = pd.read_csv("gm_2008_region.csv")
gm2008 = gm2008.drop(["Region"], axis = 1)
gm2008.head()


# In[30]:


X = gm2008.drop(["life"], axis=1)
y = gm2008["life"]


# In[31]:


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 43)


# In[35]:


reg_all = LinearRegression()
reg_all.fit(X_train, y_train)


# In[40]:


y_pred = reg_all.predict(X_test)


# In[41]:


reg_all.score(X_test, y_test)


# In[43]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse


# In[45]:


reg = LinearRegression()

cvscores_3 = cross_val_score(reg, X, y,cv = 3)
print(np.mean(cvscores_3))

cvscores_10 = cross_val_score(reg, X, y,cv = 10)
print(np.mean(cvscores_10))


# In[48]:


lasso = Lasso(alpha=0.4, normalize=True)
lasso.fit(X, y)
lasso_coef = lasso.coef_
print(lasso_coef)

