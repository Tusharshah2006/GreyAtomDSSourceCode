
# coding: utf-8

# In[1]:


import os
import pandas as pd

os.chdir('D:\GreyAtom\Datasets')


# In[3]:


ls


# In[7]:


df = pd.read_csv('spam.csv', encoding = "Latin - 1")


# In[9]:


df.head()


# In[10]:


df1 = df.copy()


# In[11]:


list(df)


# In[13]:


cols_to_be_dropped = list(df)[-3:]


# In[14]:


df = df.drop(cols_to_be_dropped, axis=1)


# In[15]:


df.head()


# In[20]:


df = df.rename(columns={"v1":"Status", "v2":"Message"})


# In[18]:


df


# In[37]:


Laptops = pd.read_csv('laptops.csv', encoding = "Latin - 1", skipinitialspace = True)


# In[38]:


Laptops.head()


# In[47]:


list(Laptops)


# In[111]:


import re
Laptops = pd.read_csv('laptops.csv', encoding = "Latin - 1", skipinitialspace = True)

Laptops.columns = [x.lower() for x in Laptops.columns]
Laptops.columns = [x.replace(" ", "_") for x in Laptops.columns]
Laptops.columns = [x.replace("(", "") for x in Laptops.columns]
Laptops.columns = [x.replace(")", "") for x in Laptops.columns]

Laptops["screen_size"] = Laptops["screen_size"].str.replace('"', '').astype(float)
Laptops["ram"] = Laptops["ram"].str.replace('GB', '').astype(int)

Laptops["weight"] = Laptops["weight"].str.replace("kg", "", regex=True)
Laptops["weight"] = Laptops["weight"].str.replace("s", "", regex=True).astype(float)

Laptops["price_euros"] = Laptops["price_euros"].str.replace(',', '.').astype(float)
#Laptops["weight"] = Laptops["weight"].str.replace("kg*", "", regex=True).astype(float)

Laptops.rename({"screen_size":"screen_size_inches", "ram":"ram_gb", "weight":"weight_kgs"}, axis=1, inplace=True)
Laptops.sort_values(['screen_size_inches', 'ram_gb'], ascending=True)

#Laptops["gpu_m"] = Laptops["gpu"].str.split(n=1,expand=True).iloc[:,0]
Laptops["gpu_m"] = Laptops.apply(lambda row: row['gpu'].split()[0], axis=1)
Laptops["cpu_m"] = Laptops.apply(lambda row: row['cpu'].split()[0], axis=1)
#Laptops["resolution"] = Laptops.apply(lambda row: row['screen'].split().str.get(-1), axis=1)
Laptops["resolution"] = Laptops["screen"].str.split().str.get(-1)

#Laptops.isnull().sum()
Laptops.to_csv('clean_laptops.csv')

