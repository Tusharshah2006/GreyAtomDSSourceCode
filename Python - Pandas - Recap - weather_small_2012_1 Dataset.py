
# coding: utf-8

# In[7]:


import pandas as pd

a = [1, 2, 3]
b = [3, 4, 5]
c = [5, 6, 7]
d = [7, 8, 9]

df2 = pd.DataFrame([a, b, c, d], index=["a", "b", "c", "d"], columns = ["x", "y", "z"])
df2.info()


# In[9]:


import os
os.chdir("D:\GreyAtom\Datasets")


# In[10]:


weather = pd.read_csv("weather_small_2012_1.csv")
weather.info()


# In[11]:


weather['Temp (C)'].unique()


# In[12]:


weather['Temp (C)'].nunique()


# In[13]:


weather = pd.read_csv("weather_small_2012.csv")
weather.info()


# In[14]:


weather.head()


# In[16]:


weather["Dew Point Temp (C)"].value_counts()


# In[19]:


ipl = pd.read_csv("ipl_matches_small.csv")
ipl.head()


# In[24]:


ipl[["toss_winner", "toss_decision"]][1:11]


# In[26]:


weather.head()


# In[29]:


v1 = weather.loc[(weather["Wind Spd (km/h)"] > 24) & (weather["Visibility (km)"] > 25)] 
v1


# In[30]:


v1 = weather.loc[(weather["Wind Spd (km/h)"] > 24) | (weather["Visibility (km)"] > 25)] 
v1

