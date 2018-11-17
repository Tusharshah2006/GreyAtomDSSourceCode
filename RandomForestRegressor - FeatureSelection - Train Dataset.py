
# coding: utf-8

# In[7]:


import os
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, classification_report, confusion_matrix, mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split as tts, cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import VotingClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, tree, DecisionTreeRegressor


# In[8]:


os.chdir("D:/GreyAtom/Datasets/09302018")
traindf = pd.read_csv("train.csv")
traindf = pd.get_dummies(traindf)


# In[9]:


not_null_train = traindf.select_dtypes(include=[np.number]).interpolate().dropna()
sum(not_null_train.isnull().sum() != 0)


# In[10]:


X = not_null_train.drop(["SalePrice"], axis=1)
y = not_null_train["SalePrice"]


# In[11]:


rf = RandomForestRegressor(n_estimators=300, random_state=2)
rf.fit(X, y)

importances = pd.Series(data=rf.feature_importances_, index = X.columns)
importances_sorted = importances.sort_values()

importances_sorted.plot(kind='barh', color="lightgreen")
plt.title('Feature Selection')
plt.show()


# In[19]:


rf = RandomForestRegressor(n_estimators=10, random_state=2)
rf.fit(X, y)

importances = pd.Series(data=rf.feature_importances_, index = X.columns)
importances_sorted = importances.sort_values()

importances_sorted.plot(kind='barh', color="lightgreen")
plt.title('Feature Selection')
plt.show()

