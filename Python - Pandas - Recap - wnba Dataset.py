
# coding: utf-8

# In[1]:


import os
os.chdir("D:\GreyAtom\Datasets")


# In[2]:


import pandas as pd
df = pd.read_csv("wnba.csv")


# In[5]:


df.head()


# In[7]:


list(df)


# In[8]:


parameter = max(df["Games Played"])


# In[9]:


parameter


# In[10]:


#Sample  is a dataframe
sample = df.sample(30, random_state=1)


# In[11]:


sample.shape


# In[12]:


statistic = max(sample["Games Played"])


# In[13]:


statistic


# In[14]:


samplling_error = parameter - statistic
print(samplling_error)


# In[18]:


population_mean = df["PTS"].mean()


# In[19]:


print(population_mean)


# In[20]:


sample = df.sample(10, random_state=1)


# In[22]:


sample_mean = sample["PTS"].mean()
print(sample_mean)


# In[26]:


sample_means = []
for i in range(100):
    sample = df.sample(10, random_state=i)
    sample_means.append(sample["PTS"].mean())


# In[27]:


sample_means


# In[35]:


sample_means = []
for i in range(100):
    sample = df.sample(142, random_state=i)
    sample_means.append(sample["PTS"].mean())

import matplotlib.pyplot as plt
plt.scatter(range(1, 101), sample_means)
plt.axhline(population_mean)
plt.show()


# In[36]:


import os
os.chdir("D:\GreyAtom\Datasets")


# In[37]:


ls


# In[38]:


import pandas as pd
df = pd.read_csv("wnba.csv")


# In[78]:


pos = pd.DataFrame(df["Pos"].value_counts())
print(pos)
heights = pd.DataFrame(df["Height"].value_counts()).sort_index(ascending=False)
heights


# In[77]:


age_ascending = pd.DataFrame((df["Age"] > 30).value_counts()).sort_index()
print(age_ascending)
age_descending = pd.DataFrame((df["Age"] < 20).value_counts()).sort_index(ascending=False)
print(age_descending)


# In[86]:


pos = pd.DataFrame(df["Pos"].value_counts())
print(pos)
print(pos/len(df)*100)
pos = df["Pos"].value_counts(normalize = True) * 100
print(pos)


# In[97]:


percentages = df["Age"].value_counts(True).sort_index() * 100
print(percentages)


# In[96]:


proportion_25 = percentages[25] / 100
print(proportion_25)


# In[98]:


df.head()


# In[100]:


df.columns


# In[107]:


games = (df["Games Played"] <= max(df["Games Played"]) / 2).value_counts(True).sort_index() * 100
print(games)


# In[111]:


from scipy.stats import percentileofscore
percentile_rank_half_less = percentileofscore(df["Games Played"], 16, kind="weak")
print(percentile_rank_half_less)


# In[113]:


df["Age"].describe()


# In[114]:


df["Age"].quantile(.25)


# In[120]:


desc = df["Age"].describe(percentiles=[.1, .2, .28]).iloc[3:]
print(desc)

