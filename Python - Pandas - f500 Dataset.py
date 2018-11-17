
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

os.chdir('D:\GreyAtom\Datasets')


# In[9]:


f500 = pd.read_csv("f500.csv")


# In[13]:


industries = f500['industry']
set(industries)


# In[15]:


previous = f500.loc[:,["rank","previous_rank", "years_on_global_500_list"]]
previous


# In[19]:


#financial_data = f500.iloc[:,2:7]
financial_data = f500.loc[:,"revenues":"profit_change"]
financial_data


# In[27]:


f500.set_index("company", inplace=True)
f500


# In[29]:


ceos = f500["ceo"]
ceos


# In[33]:


walmart = ceos["Walmart"]
walmart


# In[36]:


ceo2 = ceos["Apple":"Samsung Electronics"]
ceo2


# In[39]:


big_movers = f500.loc[["Aviva", "HP", "JD.com", "BHP Billiton"], ["rank", "previous_rank"]]
big_movers


# In[40]:


f500["country"].value_counts()


# In[42]:


max_f500 = f500.max()
max_f500


# In[44]:


f500["revenues2"] = f500["revenues"]/1000
f500


# In[56]:


country_south_korea = f500[(f500["country"]=="South Korea") | (f500["country"]=="USA")]
country_south_korea

