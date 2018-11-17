
# coding: utf-8

# In[1]:


a = [1,2,3,4,5]
b = [5,6,7,8,9]


# In[2]:


for x,y in zip(a,b):
    print (x, y)


# In[3]:


for i, x in enumerate(b):
    print(i, x)


# In[5]:


import os
os.chdir("D:\GreyAtom\Datasets\game-of-thrones")


# In[6]:


ls


# In[8]:


import pandas as pd
df = pd.read_csv("character-deaths.csv")
df.head()


# In[12]:


character_death_year = df["Death Year"]
character_death_year


# In[13]:


df.describe()


# In[14]:


df.shape


# In[17]:


people_who_died = df.loc[df["Death Year"].notnull(), ["Name", "Allegiances", "Death Year"]]
people_who_died


# In[21]:


woman_who_died = df.loc[(df["Death Year"].notnull()) & (df["Gender"]==0) , ["Name", "Allegiances", "Death Year", "Gender"]]
woman_who_died


# In[22]:


man_who_died = df.loc[(df["Death Year"].notnull()) & (df["Gender"]==1) , ["Name", "Allegiances", "Death Year", "Gender"]]
man_who_died


# In[23]:


os.chdir("D:\GreyAtom\Datasets")


# In[24]:


ls

