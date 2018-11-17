
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split as tts, cross_val_score
from sklearn.metrics import mean_squared_error


# In[4]:


import os
os.chdir("D:/GreyAtom/Datasets/09302018")

traindf = pd.read_csv("train.csv")
traindf.head()
traindf.info()


# In[27]:


correlation_values = traindf.select_dtypes(include=[np.number]).corr()
correlation_values
correlation_values[["SalePrice"]]


# In[36]:


def Threshold(ThresholdValue, df):
    return df[(df["SalePrice"] >= ThresholdValue) | (df["SalePrice"] <= ThresholdValue*-1)].index
                
colName = Threshold(0.6, correlation_values)
print(colName)


# In[45]:


correlation_values = traindf[colName].select_dtypes(include=[np.number]).corr()
correlation_values


# In[46]:


#Model 1 - Features - OverallQual, TotalBsmtSF, GrLivArea, GarageArea


# In[53]:


X = traindf[["OverallQual", "TotalBsmtSF", "GrLivArea", "GarageArea", "GarageCars", "1stFlrSF"]]
y = traindf["SalePrice"]

X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)

LinReg = LinearRegression()
LinReg.fit(X_train, y_train)  

y_pred = LinReg.predict(X_test)
rsquare = LinReg.score(X_test, y_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rsquare, rmse)

