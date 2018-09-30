
# coding: utf-8

# In[1]:


import os
os.chdir("D:\GreyAtom\Datasets")


# In[2]:


ls


# In[13]:


import pandas as pd
df = pd.read_excel("populations_year.xlsx")
df.head()


# In[5]:


import matplotlib.pyplot as plt


# In[14]:


pop = df['Population']
year = df['Year']


# In[15]:


plt.plot(pop, year)


# In[16]:


gdp_cap = df['gdp_cap']
life_exp = df['life_exp']


# In[17]:


plt.scatter(gdp_cap, life_exp)


# In[20]:


xlab = "GDP per Capita in USD"
ylab = "Life Expectancy in years"
title = "World Development in 2007"
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)
plt.scatter(gdp_cap, life_exp)


# In[22]:


xlab = "GDP per Capita in USD"
ylab = "Life Expectancy in years"
title = "World Development in 2007"
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)

tick_val = [1000, 10000, 20000, 30000, 40000, 50000]
tick_lab = ["1K", "10K", "20K", "30K", "40K", "50K"]
plt.xticks(tick_val, tick_lab)

tick_val = [40, 50, 60, 70, 80, 90]
tick_lab = ["4", "5", "6", "7", "8", "9"]
plt.yticks(tick_val, tick_lab)

plt.scatter(gdp_cap, life_exp)


# In[23]:


import numpy as np
pop


# In[32]:


np_pop = np.array(pop*6)


# In[33]:


xlab = "GDP per Capita in USD"
ylab = "Life Expectancy in years"
title = "World Development in 2007"
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)

tick_val = [1000, 10000, 20000, 30000, 40000, 50000]
tick_lab = ["1K", "10K", "20K", "30K", "40K", "50K"]
plt.xticks(tick_val, tick_lab)

tick_val = [40, 50, 60, 70, 80, 90]
tick_lab = ["4", "5", "6", "7", "8", "9"]
plt.yticks(tick_val, tick_lab)

plt.scatter(gdp_cap, life_exp,s=np_pop)


# In[35]:


plt.hist(np_pop, bins=100)

