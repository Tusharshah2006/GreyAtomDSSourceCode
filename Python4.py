
# coding: utf-8

# In[1]:


ls


# In[3]:


import os
os.chdir("D:\GreyAtom\Datasets")


# In[5]:


names=["United States", "Australia", "Japan", "India", "Russia", "Morocco", "Egypt"]
dr = [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
my_dict = {}


# In[8]:


my_dict["countries"] = names
my_dict["drive_right"] = dr
my_dict["cars_per_capita"] = cpc


# In[9]:


my_dict


# In[12]:


import pandas as pd
cars = pd.DataFrame(my_dict)
cars


# In[13]:


row_labels = ["USA", "AUS", "JAP", "IND", "RUS", "MOR", "ENG"]


# In[14]:


cars.index = row_labels
cars


# In[22]:


#This is an example of series
country = cars["countries"]
country
type(country)


# In[23]:


#This is an example of Data Frame
country = cars[["countries", "cars_per_capita"]]
country
type(country)


# In[26]:


cars = pd.read_csv("cars.csv", index_col=0)


# In[27]:


cars


# In[30]:


cars[["cars_per_cap", "drives_right"]]


# In[31]:


cars[:4]


# In[32]:


cars


# In[35]:


cars[2::3]


# In[37]:


cars.loc[["JAP", "MOR", "RU", "IN"]]


# In[39]:


cars.ix[[3, 2]]

