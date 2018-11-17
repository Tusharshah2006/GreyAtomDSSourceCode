
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split as tts, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer


# In[40]:


import os
os.chdir("D:/GreyAtom/Datasets/09302018")

traindf = pd.read_csv("Train_UWu5bXk.csv")
traindf.head()
traindf.info()


# In[41]:


imp = Imputer(missing_values = float("NaN"), strategy="mean")
traindf["Item_Weight"] = imp.fit_transform(traindf[["Item_Weight"]]).ravel()


# In[42]:


X = traindf[["Item_MRP", "Outlet_Establishment_Year", "Item_Visibility", "Item_Weight"]]
y = traindf["Item_Outlet_Sales"]


# In[43]:


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)

LinReg = LinearRegression()
LinReg.fit(X_train, y_train)


# In[44]:


y_pred = LinReg.predict(X_test)
rsquare = LinReg.score(X_test, y_test)
rsquare


# In[45]:


coeff = pd.DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = pd.Series(LinReg.coef_)
coeff


# In[46]:


plot.bar(coeff[0], coeff['Coefficient Estimate'])

