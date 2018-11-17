
# coding: utf-8

# In[1]:


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
from sklearn.tree import DecisionTreeClassifier, tree, DecisionTreeClassifier 


# In[8]:


os.chdir("D:/GreyAtom/Datasets")
df = pd.read_csv("indian_liver_patient_preprocessed.csv")
#traindf = pd.get_dummies(traindf)
df.head()


# In[9]:


df.drop("Unnamed: 0", axis=1, inplace=True)


# In[11]:


X = df.drop("Liver_disease", axis=1)
y = df["Liver_disease"]


# In[31]:


dtc = DecisionTreeClassifier(random_state=42)
print(dtc.get_params())


# In[32]:


dt = DecisionTreeClassifier(max_depth=9, criterion='gini', max_features=0.6, min_samples_leaf=0.16)
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

score = dt.score(X_test, y_test)
acc = accuracy_score(y_test, y_pred)
print(score, acc)


# In[20]:


print(classification_report(y_test, y_pred))


# In[28]:


#Create a dictionary of possible parameters
params = {'max_depth':np.arange(1, 10), 
                'max_features':np.arange(0.2, 0.8, 0.1), 
                'min_samples_leaf':np.arange(0.05, 0.2, 0.01)}

#Create the GridSearchCV object
grid_clf = GridSearchCV(estimator=dtc, param_grid=params, cv=10,n_jobs=-1,scoring="accuracy")

#Fit the data with the best possible parameters
grid_clf = grid_clf.fit(X, y)

#Print the best estimator with it's parameters
print (grid_clf.best_estimator_)
print(grid_clf.best_score_)


# In[29]:


grid_clf.best_params_


# In[30]:


rf = RandomForestRegressor(n_estimators=2500, random_state=2)
rf.fit(X, y)

importances = pd.Series(data=rf.feature_importances_, index = X.columns)
importances_sorted = importances.sort_values()

importances_sorted.plot(kind='barh', color="lightgreen")
plt.title('Feature Selection')
plt.show()

