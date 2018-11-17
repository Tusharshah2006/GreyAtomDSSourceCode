
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')


# # Load data and get familiar with it

# ## Data properties

# In[2]:


train = pd.read_csv('train.csv')
print(train.shape)
print(train.dtypes)


# ## Sample data

# In[3]:


train.head()


# In[4]:


train.tail()


# In[5]:


train.describe()


# In[6]:


train.sort_values(by='LotArea')


# ## Plotting

# In[7]:


plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


# In[8]:


plt.hist(train.YrSold)
plt.show()


# # Target

# In[9]:


plt.hist(train.SalePrice)
plt.show()


# In[10]:


target = np.log10(train.SalePrice)
plt.hist(target)
plt.show()


# # Features

# ## Handling categorical data

# In[11]:


train['LandContour']


# In[12]:


train['CategoricalLandContour'] = train['LandContour'].astype("category")
train['CategoricalLandContour']


# In[13]:


categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()


# ## Correlation between features

# In[14]:


correlation_values = train.select_dtypes(include=[np.number]).corr()
correlation_values


# In[15]:


f, ax = plt.subplots(figsize=(15, 12))
_ = sns.heatmap(correlation_values, linecolor = 'white', cmap = 'magma', linewidths = 1)


# ## Relationship between features

# In[16]:


plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()


# ### Removing outliers

# In[17]:


train = train[train['GarageArea'] < 1200]
plt.scatter(x=train['GarageArea'], y=np.log10(train.SalePrice))
plt.xlim(-200,1600) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


# ## Null values

# In[18]:


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls = nulls[nulls>0]
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls


# ## Missing data

# In[19]:


train.isnull()


# In[20]:


train.isnull().sum()


# In[21]:


train.isnull().sum() != 0


# In[22]:


sum(train.isnull().sum() != 0)


# In[23]:


not_null_train = train.select_dtypes(include=[np.number]).interpolate().dropna()
sum(not_null_train.isnull().sum() != 0)


# # Training and Evaluation (Base model)

# In[28]:


from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# In[31]:


def train_and_evaluate(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
    lr =LinearRegression()
    my_model = lr.fit(X_train, y_train)
    
    train_predictions = my_model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    print('train error: ', train_rmse)
    
    predictions = my_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print('test_error: ', test_rmse)
    
    sns.regplot(y = train_predictions, x = y_train, color = 'red', label = 'Training Data', scatter_kws={'alpha':0.75})
    sns.regplot(y = predictions, x = y_test, color = 'green', label = 'Validation Data', scatter_kws={'alpha':0.75})
    plt.title('Predicted Values vs Test Values')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.legend(loc = 'upper left')
    


# In[32]:


y1 = np.log10(train.SalePrice)
X1 = not_null_train.drop(['SalePrice', 'Id'], axis=1)
train_and_evaluate(X1, y1)


# #  How to improve results further?

# In[33]:


train = pd.read_csv('train.csv')
train.dtypes


# ## Fill missing data

# In[34]:


# categorical
print(train['PoolQC'])
train['PoolQC'].value_counts()
train['PoolQC'].fillna('No Pool', inplace = True)
print(train['PoolQC'])


# In[35]:


train['MiscFeature'].fillna('None', inplace = True)
train['MiscFeature'].value_counts()


# In[36]:


train['Alley'].fillna('No alley access', inplace = True)
train['Fence'].fillna('No Fence', inplace = True)
train['FireplaceQu'].fillna('No Fireplace', inplace = True)
train['GarageType'].fillna('No Garage', inplace = True)
train['GarageCond'].fillna('No Garage', inplace = True)
train['GarageFinish'].fillna('No Garage', inplace = True)
train['GarageQual'].fillna('No Garage', inplace = True)
train['BsmtFinType2'].fillna('No Basement', inplace = True)
train['BsmtExposure'].fillna('No Basement', inplace = True)
train['BsmtQual'].fillna('No Basement', inplace = True)
train['BsmtCond'].fillna('No Basement', inplace = True)
train['BsmtFinType1'].fillna('No Basement', inplace = True)
train['MasVnrType'].fillna('None', inplace = True)
train['GarageYrBlt'].fillna('No Garage', inplace = True)


# In[37]:


# Fill with Mode
print(train['Electrical'].value_counts())
train['Electrical'].fillna(value = 'SBrkr', inplace = True)
train['Electrical'].value_counts()


# In[38]:


# Fill with 0
train['MasVnrArea'].fillna(0, inplace = True)
train['LotFrontage'].fillna(0, inplace = True)


# In[39]:


cat_vars = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'YrSold', 'MoSold', 
       'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'YearBuilt', 'YearRemodAdd']
for col in cat_vars:
    train[col] = train[col].astype('category',copy=False)

train.dtypes


# ## Dummy variables for categorical data

# In[40]:


# Creating Dummy Variables
cat = pd.get_dummies(train[cat_vars], drop_first = True)
train_dummy = train
train_dummy = pd.concat([train_dummy, cat], axis = 1)
train_dummy.drop(cat_vars, axis = 1, inplace = True)
train_dummy.dtypes


# ## Evaluation

# In[41]:


y = np.log10(train.SalePrice)
X = train_dummy.drop(['SalePrice', 'Id'], axis=1)
train_and_evaluate(X, y)


# ## Check for correlated variables

# In[42]:


correlations = train.corr()

f, ax = plt.subplots(figsize=(15, 12))
_ = sns.heatmap(correlation_values, linecolor = 'white', cmap = 'magma', linewidths = 1)


# In[43]:


correlations = correlations.iloc[:36, :36] 
cut_off = 0.5
high_corrs = correlations[correlations.abs() > cut_off][correlations.abs() != 1].unstack().dropna().to_dict()
high_corrs = pd.Series(high_corrs, index = high_corrs.keys())
high_corrs = high_corrs.reset_index()
high_corrs = pd.DataFrame(high_corrs)
high_corrs.columns = ['Attributes', 'Correlations']
high_corrs['Correlations'] = high_corrs['Correlations'].drop_duplicates(keep = 'first')
high_corrs.dropna().sort_values(by = 'Correlations', ascending = False)


# In[44]:


mvp_list = ['GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 
            'TotRmsAbvGrd']
num_vars = ['GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 
            'TotRmsAbvGrd', 'SalePrice']


# ## Variables correlated with target

# In[46]:


k = 10
cols = correlations.nlargest(k, 'SalePrice')['SalePrice'].index
cols
# cm = np.corrcoef(train[cols].values.T)
# sns.set(font_scale=1.25)
# f, ax = plt.subplots(figsize=(12, 9))
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, linewidth = 5,
#                  yticklabels=cols.values, xticklabels=cols.values, cmap = 'viridis', linecolor = 'white')
# plt.show()


# ## Dummying categorical data

# In[47]:


# Selected categorical variables
cat_mvp_vars = ['MSSubClass', 'MSZoning', 'Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 
                'OverallQual', 'OverallCond', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'HeatingQC', 'CentralAir',
                'KitchenQual', 'GarageFinish', 'GarageQual', 'PoolQC', 'SaleType', 'YearRemodAdd']


# In[48]:


d = train.iloc[:, 10:11]
print(d.dtypes)
d.LotConfig.unique


# In[49]:


dummies = pd.get_dummies(d)
print(dummies.dtypes)
print(dummies)


# ## Feature selection

# In[50]:


best_features = mvp_list + cat_mvp_vars

categorical = pd.get_dummies(train[cat_mvp_vars], drop_first = True)

dummy_best_feature_train = train[num_vars]
dummy_best_feature_train = pd.concat([dummy_best_feature_train, categorical], axis = 1)
dummy_best_feature_train.dtypes


# In[51]:


y = np.log10(train.SalePrice)
X = dummy_best_feature_train.drop(['SalePrice'], axis=1)
train_and_evaluate(X, y)

