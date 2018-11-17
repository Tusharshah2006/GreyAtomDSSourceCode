
# coding: utf-8

# In[91]:


import os


# In[92]:


os.getcwd()


# In[93]:


ls


# In[94]:


import pandas as pd


# In[95]:


df= pd.read_csv("dc_airbnb.csv")


# In[96]:


from sklearn.preprocessing import Imputer


# In[97]:


numeric_df = df._get_numeric_data()


# In[98]:


numeric_cols = list(numeric_df)


# In[99]:


imp_numeric = Imputer(missing_values="NaN",
                     strategy = "mean",
                     axis = 1)


# In[100]:


imp_numeric.fit(numeric_df)


# In[101]:


numeric_df = imp_numeric.transform(numeric_df)


# In[102]:


numeric_df = pd.DataFrame(numeric_df)


# In[103]:


numeric_df.columns = numeric_cols


# In[104]:


numeric_df


# In[105]:


numeric_df.isnull().sum()


# In[106]:


import numpy as np
categorical_cols = df.select_dtypes(exclude = np.number)


# In[107]:


categorical_cols.shape


# In[108]:


categorical_cols.isnull().sum()


# In[109]:


categorical_cols = categorical_cols.drop(["cleaning_fee",
                                         "security_deposit",
                                         "zipcode"],1)


# In[110]:


categorical_cols["host_acceptance_rate"] = categorical_cols["host_acceptance_rate"].str.replace('%','').astype(float)


# In[111]:


categorical_cols["host_response_rate"] = categorical_cols["host_response_rate"].str.replace('%','').astype(float)


# In[112]:


categorical_cols["price"] = categorical_cols["price"].str.replace(',','')
categorical_cols["price"] = categorical_cols["price"].str.replace('$','').astype(float)


# In[113]:


numeric_df["host_response_rate"] = categorical_cols["host_response_rate"]


# In[114]:


numeric_df["host_acceptance_rate"] = categorical_cols["host_acceptance_rate"]


# In[115]:


numeric_df["price"] = categorical_cols["price"]


# In[116]:


categorical_cols = categorical_cols.drop(["host_acceptance_rate",
                                         "host_response_rate",
                                          "price"],1)


# In[117]:


final_df = pd.concat([numeric_df,categorical_cols],1)


# In[118]:


final_df.isnull().sum()


# In[119]:


final_df["host_acceptance_rate"].replace(np.nan,final_df["host_acceptance_rate"].mean()
                                         ,inplace = True)


# In[120]:


final_df["host_response_rate"].replace(np.nan,final_df["host_response_rate"].mean(),
                                         inplace = True)


# In[121]:


final_df.isnull().sum()


# In[122]:


final_df.head(2)


# In[123]:


X = final_df.drop(["price"],1)


# In[124]:


y = final_df["price"]


# In[136]:


from sklearn.ensemble import RandomForestRegressor
seed = 42
import matplotlib.pyplot as plt
X = pd.get_dummies(X)
rf = RandomForestRegressor(n_estimators=3000,
                           min_samples_leaf=0.08,
                          random_state = seed)


# In[137]:


rf.fit(X,y)


# In[138]:


importances_rf = pd.Series(rf.feature_importances_,
                          index = X.columns)
sorted_importances = importances_rf.sort_values()


# In[139]:


sorted_importances.plot(kind = "barh",
                       color = "blue")


# In[134]:


df["room_type"].value_counts()


# In[135]:


list(final_df)


# In[145]:


final_df = pd.get_dummies(final_df)


# In[146]:


corr = final_df.corr()


# In[149]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
final_df=pd.DataFrame(scaler.fit_transform(final_df),
                      columns = final_df.columns)


# In[152]:


corr= final_df.corr()
corr["price"].sort_values()


# In[153]:


lin_reg_features = ["accomodates","room_type_Entire home/apt", 
           "host_listings_count", "room_type_Private room",
                   "price"]


# In[167]:


lin_df = final_df[["room_type_Private room","accommodates", "price", "room_type_Entire home/apt", "host_listings_count"]]


# In[170]:


from sklearn.model_selection import train_test_split as tts
X= lin_df.drop(["price"],1)
y = lin_df["price"]
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
X_train,X_test,y_train,y_test = tts(X,y,random_state = seed,
                                   test_size = 0.2)


# In[171]:


lin_reg.fit(X_train,y_train)


# In[172]:


y_pred = lin_reg.predict(X_test)


# In[173]:


lin_reg.score(X_test,y_test)

