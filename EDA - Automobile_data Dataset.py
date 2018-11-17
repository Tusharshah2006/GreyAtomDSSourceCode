
# coding: utf-8

# In[80]:


import os
import pandas as pd
os.chdir("D:\GreyAtom\Datasets")


# In[81]:


autoData = pd.read_csv("Automobile_data.csv")
autoData_Copy = autoData.copy() 


# In[82]:


autoData.shape


# In[83]:


autoData.columns


# In[84]:


autoData.dtypes


# In[85]:


autoData.info()


# In[86]:


autoData.describe()


# In[87]:


autoData["normalized-losses"].value_counts()


# In[88]:


normalized_losses_filter = autoData[autoData["normalized-losses"] != "?"]
meanValue = normalized_losses_filter["normalized-losses"].astype(float).mean()


# In[89]:


autoData["normalized-losses"] = autoData["normalized-losses"].str.replace('?', str(meanValue)).astype(float)


# In[90]:


autoData["normalized-losses"].describe()


# In[91]:


autoData["price"].value_counts()


# In[92]:


price_filter = autoData["price"].loc[autoData["price"] != "?"]
autoData["price"] = autoData["price"].str.replace('?', str(price_filter.astype(float).median())).astype(float)


# In[93]:


horsepower_filter = autoData["horsepower"].loc[autoData["horsepower"] != "?"]
autoData["horsepower"] = autoData["horsepower"].str.replace('?', str(horsepower_filter.astype(float).median())).astype(float)


# In[94]:


peak_rpm_filter = autoData["peak-rpm"].loc[autoData["peak-rpm"] != "?"]
autoData["peak-rpm"] = autoData["peak-rpm"].str.replace('?', str(peak_rpm_filter.astype(float).mean())).astype(float)


# In[95]:


autoData["bore"] = pd.to_numeric(autoData["bore"], errors = "coerce")


# In[96]:


stroke_filter = autoData["stroke"].loc[autoData["stroke"] != "?"]
autoData["stroke"] = autoData["stroke"].str.replace('?', str(stroke_filter.astype(float).mean())).astype(float)


# In[97]:


no_of_doors_filter = autoData["num-of-doors"].loc[autoData["num-of-doors"] != "?"]
autoData["num-of-doors"] = autoData["num-of-doors"].str.replace('?', str(no_of_doors_filter.mode()))


# In[99]:


autoData["bore"].fillna(autoData["bore"].median(), inplace = True)


# In[100]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[101]:


autoData["make"].value_counts().plot(kind = "bar")


# In[102]:


autoData["make"].value_counts().nlargest(5).plot(kind = "bar")


# In[103]:


autoData["symboling"].hist(bins = 6, color = "blue")


# In[104]:


autoData["normalized-losses"].plot(kind = "bar")


# In[105]:


autoData["normalized-losses"].hist(bins = 16, color = "blue")


# In[110]:


autoData["fuel-type"].value_counts().plot(kind = "bar")


# In[111]:


autoData["fuel-type"].value_counts().plot.pie(autopct = "%.2f")


# In[112]:


autoData["aspiration"].value_counts().plot.pie(autopct = "%.2f")


# In[113]:


autoData["drive-wheels"].value_counts().plot.pie(autopct = "%.2f")


# In[116]:


autoData["horsepower"].value_counts().hist(bins=6)
plt.title("Horse Power Graph")
plt.ylabel("Number of V3hicles")
plt.xlabel("HP")


# In[119]:


autoData["num-of-doors"].value_counts().plot(kind = "bar")


# In[120]:


autoData["engine-size"].value_counts().hist(bins=6)


# In[123]:


corr = autoData.corr()
corr


# In[130]:


sns.set_context("notebook", font_scale = 1.0, rc = {"lines.linewidth":2.5})
plt.figure(figsize = (13, 7))
a = sns.heatmap(corr, annot = True, fmt = ".2f")


# In[132]:


g = sns.lmplot("price", "engine-size", autoData)


# In[133]:


g = sns.lmplot("normalized-losses", "symboling", autoData)


# In[135]:


plt.scatter(autoData["engine-size"], autoData["peak-rpm"])


# In[139]:


g = sns.lmplot("city-mpg", "curb-weight", autoData, hue = "make", fit_reg = False)


# In[140]:


g = sns.lmplot("highway-mpg", "curb-weight", autoData, hue = "make", fit_reg = False)


# In[143]:


plt.rcParams['figure.figsize'] = (10, 5)
ax = sns.boxplot(x = "drive-wheels", y = "price", data = autoData)


# In[144]:


plt.rcParams['figure.figsize'] = (23, 10)
ax = sns.boxplot(x = "make", y = "price", data = autoData)

