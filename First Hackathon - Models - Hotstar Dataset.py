
# coding: utf-8

# In[1]:


import os 
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


# In[2]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
# from xgboost import XGBClassifier


# In[3]:


from sklearn.metrics import classification_report, accuracy_score,auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import  LogisticRegression
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
import operator
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# In[4]:


# os.chdir("D:/GreyAtom/Hackathon/hotstar_dataset")
df = pd.read_json("train_data.json")


# In[5]:


dfT = df.T
dfT.head()


# In[6]:


dfT["titles"]=dfT["titles"].str.split(",")
dfT["titles_count"]=dfT["titles"].apply(lambda x:len(x))


# In[7]:


dfT.head()


# In[8]:


Metros = ["mumbai", "gurgaon", "kolkata", "delhi", "chennai", "bangalore", "pune", "hyderabad", "new delhi", "delhi", "navi mumbai"]

dfT["city_only"] = dfT["cities"].apply(lambda x: [i.rsplit(":")[0] for i in x.rsplit(",")])
dfT["metro"] = dfT["city_only"].apply(lambda x: sum([1 for i in x if i in Metros]))
dfT['non_metro'] = dfT['city_only'].apply(lambda x: sum([1 for i in x if i not in Metros]))


# In[9]:


dfT.head()


# In[10]:


dfT["final_hour"] = dfT["tod"].apply(lambda x: [i.rsplit(":")[0] for i in x.rsplit(",")])

dfT["Morning Viewer"] = dfT["final_hour"].apply(lambda x: sum([1 for i in x if int(i)<=12]))
dfT["Afternoon Viewers"] = dfT["final_hour"].apply(lambda x: sum([1 for i in x if int(i)>12 and int(i)<17]))
dfT["Evening Viewers"] = dfT["final_hour"].apply(lambda x:sum([1 for i in x if int(i)>=17 and int(i)<20]))
dfT["Night Viewers"] = dfT["final_hour"].apply(lambda x: sum([1 for i in x if int(i)>=20 and int(i)<=24]))


# In[11]:


dfT.head()


# In[12]:


weekdays = [2, 3, 4, 5, 6]
dfT["dow_list"] = dfT["dow"].apply(lambda x: [i.rsplit(":")[0] for i in x.rsplit(",")])

# dfT["Sunday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 1]))
# dfT["Monday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 2]))
# dfT["Tuesday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 3]))
# dfT["Wednesday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 4]))
# dfT["Thursday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 5]))
# dfT["Friday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 6]))
# dfT["Saturday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 7]))

dfT["Weekday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) in weekdays]))
dfT["Weekend"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) not in weekdays]))


# In[13]:


dfT.head()


# In[14]:


Sports = ['Cricket', 'Football', 'Badminton', 'Sport', 'Formula1', 'Hockey', 'Kabaddi', 'Table Tennis', 'Tennis', 'Volleyball', 'Athletics', 'FormulaE', 'Boxing', 'Swimming', 'IndiaVsSa']
Television = ['Wildlife', 'LiveTV', 'TalkShow', 'Reality', 'Awards', 'Travel', 'Science', 'Documentary', 'Mythology', 'Kids']
Movies = ['Drama', 'Family', 'Crime', 'Romance', 'Action', 'Comedy', 'Thriller', 'Teen', 'Horror', 'NA']

dfT["genres_list"] = dfT["genres"].apply(lambda x: [i.rsplit(":")[0] for i in x.rsplit(",")])

# genres_count = {}
# for i in dfT['genres']:
#     for j in (i.split(',')):
#         if j.split(':')[0] not in genres_count:
#              genres_count[j.split(':')[0]]= 1
#         else:
#              genres_count[j.split(':')[0]] += 1

# for key, value in genres_count.items():
#     dfT[key] = ""
#     dfT[key] = dfT["genres_list"].apply(lambda x: sum([1 for i in x if i == key]))

dfT['Sports'] = dfT["genres_list"].apply(lambda x: sum([1 for i in x if i in Sports]))
dfT['Television'] = dfT["genres_list"].apply(lambda x: sum([1 for i in x if i in Television]))
dfT['Movies'] = dfT["genres_list"].apply(lambda x: sum([1 for i in x if i in Movies]))


# In[15]:


dfT.head()


# In[16]:


dfT["viewing_time_only"]=dfT["tod"].apply(lambda x: [i.split(":")[1] for i in x.split(",")])
dfT["Summed_time"]=dfT["viewing_time_only"].apply(lambda x: sum([int(i) for i in x ]))


# In[17]:


dfT.head()


# In[18]:


dfT = dfT.drop(['cities'],1)


# In[43]:


dfT=dfT.drop(['dow','genres','titles','tod','city_only','final_hour','viewing_time_only','dow_list'],1)


# In[46]:


dfT.drop(['genres_list'],1)


# In[49]:


dfT=dfT.drop(['genres_list'],1)


# In[51]:


dfT.dtypes


# In[52]:


dfT.head()


# In[53]:


dfT['segment']=dfT['segment'].replace('neg',0).replace('pos',1)


# In[54]:


dfT.head()


# In[55]:


list(dfT)


# In[56]:


X=dfT.drop(['segment'],1)
y=dfT['segment']


# In[57]:


dfT['segment'].loc[dfT['Summed_time']>(dfT['Summed_time'].quantile(0.75)-dfT['Summed_time'].quantile(0.25))*1.5+dfT['Summed_time'].quantile(0.75)]


# In[58]:


from imblearn.over_sampling import SMOTE


# In[59]:


from sklearn.model_selection import train_test_split as tts


# In[60]:


X_train,X_test,y_train,y_test = tts(X,y,random_state=42,test_size=0.3)


# In[61]:


sm = SMOTE()


# In[62]:


x_smote,y_smote=sm.fit_sample(X_train,y_train)


# In[63]:


x_smote=pd.DataFrame(x_smote)

y_smote=pd.DataFrame(y_smote)

y_smote[0].value_counts()


# In[64]:


(x_smote.columns)=list(X_train.columns)
X_train


# In[68]:


df_smote = pd.concat([x_smote,y_smote],1)

df_smote=df_smote.rename(index=str, columns={0: "segment"})

df_smote


# In[69]:


X1 = df_smote.drop(['segment'],1)
y1 = df_smote['segment']


# In[93]:


# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)


clf = GridSearchCV(log_reg, hyperparameters, cv=5, verbose=0)
# Fit grid search
best_model = clf.fit(X1, y1)


# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# Log_reg on X_test

# In[77]:


log_reg = LogisticRegression(C=2.7825594022071245)
X1_train,X1_test,y1_train,y1_test = tts(X1,y1,random_state=42,test_size=0.3)
model_log_reg = log_reg.fit(X1_train,y1_train)
y_pred_log_reg = model_log_reg.predict(X_test)
y_pred_proba_log_reg = model_log_reg.predict_proba(X_test)[:,1]



print(accuracy_score(y_test,y_pred_log_reg))
print(classification_report(y_test,y_pred_log_reg))

from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds = roc_curve(y_test,y_pred_proba_log_reg)
print(roc_auc_score(y_test,y_pred_proba_log_reg))# Iterate over probabiliteis and not y_pred


# LogReg on X_train

# In[80]:


log_reg = LogisticRegression(C=2.7825594022071245)
X1_train,X1_test,y1_train,y1_test = tts(X1,y1,random_state=42,test_size=0.3)
model_log_reg = log_reg.fit(X1_train,y1_train)
y_pred_log_reg = model_log_reg.predict(X1_train)
y_pred_proba_log_reg = model_log_reg.predict_proba(X1_train)[:,1]



print(accuracy_score(y1_train,y_pred_log_reg))
print(classification_report(y1_train,y_pred_log_reg))

from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds = roc_curve(y1_train,y_pred_proba_log_reg)
print(roc_auc_score(y1_train,y_pred_proba_log_reg))# Iterate over probabiliteis and not y_pred


# In[ ]:







# NaiveBayes on X_test

# In[82]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
model_nb = nb.fit(X1_train,y1_train)
y_pred_nb  = model_nb.predict(X_test)
y_pred_proba_nb = model_nb.predict_proba(X_test)[:,1]

print(accuracy_score(y_test,y_pred_nb))
print(classification_report(y_test,y_pred_nb))


from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds = roc_curve(y_test,y_pred_proba_nb)
roc_auc_score(y_test,y_pred_proba_nb)# Iterate over probabiliteis and not y_pred


# NaiveBayes on X_train

# In[84]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
model_nb = nb.fit(X1_train,y1_train)
y_pred_nb  = model_nb.predict(X1_train)
y_pred_proba_nb = model_nb.predict_proba(X1_train)[:,1]

print(accuracy_score(y1_train,y_pred_nb))
print(classification_report(y1_train,y_pred_nb))


from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds = roc_curve(y1_train,y_pred_proba_nb)
roc_auc_score(y1_train,y_pred_proba_nb)# Iterate over probabiliteis and not y_pred


# Dtc on X_test

# In[85]:


from sklearn.tree import DecisionTreeClassifier
seed =42
dtc = DecisionTreeClassifier(random_state=seed,criterion='gini')


# In[86]:


params = {"max_depth":np.arange(2,4),
         "min_samples_leaf":np.arange(0.05,0.1,0.01),
         "max_features":np.arange(0.2,0.5,0.1)}

grid_dt = GridSearchCV(estimator=dtc,param_grid=params,
                      cv=10,n_jobs=1,#n_jobs=-1 will run all core processors on gpu
                      scoring="accuracy")

model_dtc=dtc.fit(X1_train,y1_train)



y_pred_dtc=model_dtc.predict(X_test)
print(accuracy_score(y_test,y_pred_dtc))
print(classification_report(y_test,y_pred_dtc))

y_pred_proba_dtc = model_dtc.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds = roc_curve(y_test,y_pred_proba_dtc)
roc_auc_score(y_test,y_pred_proba_dtc)# Iterate over probabiliteis and not y_pred


# dtc on X_train

# In[87]:


model_dtc=dtc.fit(X1_train,y1_train)



y_pred_dtc=model_dtc.predict(X1_train)
print(accuracy_score(y1_train,y_pred_dtc))
print(classification_report(y1_train,y_pred_dtc))

y_pred_proba_dtc = model_dtc.predict_proba(X1_train)[:,1]

from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds = roc_curve(y1_train,y_pred_proba_dtc)
print(roc_auc_score(y1_train,y_pred_proba_dtc))# Iterate over probabiliteis and not y_pred


# XGBoost predict on X_test

# In[88]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
model_xgb=xgb.fit(X1_train, y1_train)
y_pred_xgb = model_xgb.predict(X_test)
print(accuracy_score(y_test,y_pred_xgb))
print(classification_report(y_test,y_pred_xgb))

y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds = roc_curve(y_test,y_pred_proba_xgb)
print(roc_auc_score(y_test,y_pred_proba_xgb))# Iterate over probabiliteis and not y_pred


# XGBoost on X_train

# In[89]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
model_xgb=xgb.fit(X1_train, y1_train)
y_pred_xgb = model_xgb.predict(X1_train)
print(accuracy_score(y1_train,y_pred_xgb))
print(classification_report(y1_train,y_pred_xgb))

y_pred_proba_xgb = model_xgb.predict_proba(X1_train)[:,1]
from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds = roc_curve(y1_train,y_pred_proba_xgb)
print(roc_auc_score(y1_train,y_pred_proba_xgb))# Iterate over probabiliteis and not y_pred


# RFC on X_test

# In[91]:


rfc = RandomForestClassifier()
model_rfc = rfc.fit(X1_train,y1_train)
y_pred_rfc = model_rfc.predict(X_test)
print(accuracy_score(y_test,y_pred_rfc))
print(classification_report(y_test,y_pred_rfc))

y_pred_proba_rfc = model_dtc.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds = roc_curve(y_test,y_pred_proba_rfc)
roc_auc_score(y_test,y_pred_proba_rfc)# Iterate over probabiliteis and not y_pred


# RFC on X_train

# In[92]:


rfc = RandomForestClassifier()
model_rfc = rfc.fit(X1_train,y1_train)
y_pred_rfc = model_rfc.predict(X1_train)
print(accuracy_score(y1_train,y_pred_rfc))
print(classification_report(y1_train,y_pred_rfc))

y_pred_proba_rfc = model_dtc.predict_proba(X1_train)[:,1]
from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds = roc_curve(y1_train,y_pred_proba_rfc)
roc_auc_score(y1_train,y_pred_proba_rfc)# Iterate over probabiliteis and not y_pred

