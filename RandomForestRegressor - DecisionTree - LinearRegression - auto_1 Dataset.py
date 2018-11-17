
# coding: utf-8

# In[36]:


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


# In[7]:


os.chdir("D:/GreyAtom/Datasets")
df = pd.read_csv("auto_1.csv")


# In[9]:


df.head()


# In[12]:


df = pd.get_dummies(df)

X = df.drop(["mpg"], axis=1)
y = df["mpg"]


# In[26]:


rf = RandomForestRegressor(n_estimators=2500, random_state=2)
rf.fit(X, y)


# In[27]:


importances = pd.Series(data=rf.feature_importances_, index = X.columns)
importances_sorted = importances.sort_values()


# In[28]:


importances_sorted.plot(kind='barh', color="lightgreen")
plt.title('Feature Selection')
plt.show()


# In[29]:


plt.scatter(df["mpg"], df["hp"])


# In[46]:


X = df[['displ', 'weight', 'size', 'hp']]
y = df["mpg"]


# In[51]:


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)

LinReg = LinearRegression()
LinReg.fit(X_train, y_train)  

y_pred = LinReg.predict(X_test)
rsquare = LinReg.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rsquare, rmse, mse)


# In[52]:


dt = DecisionTreeRegressor(max_depth=9, criterion="mse", min_samples_leaf=0.16)
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

score = dt.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print(score, mse)

