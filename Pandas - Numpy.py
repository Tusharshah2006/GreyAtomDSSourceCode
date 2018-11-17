
# coding: utf-8

# In[4]:


import os
import numpy as np
import pandas as pd
os.chdir ('D:\GreyAtom\Datasets')


# In[9]:


taxis_data_np = np.genfromtxt("nyc_taxis.csv", delimiter=',', skip_header=1)
taxi_data_pd = pd.read_csv("nyc_taxis.csv")


# In[10]:


taxi_ten_np = taxis_data_np[:10]


# In[11]:


taxi_ten_np


# In[12]:


rows_391_to_500 = taxis_data_np[391:500]


# In[14]:


rows_32_to_78_columns_5 = taxis_data_np[32:78,5]


# In[15]:


trip_distance_miles = taxis_data_np[:,7]
trip_length_seconds = taxis_data_np[:,8]
trip_length_hours = trip_length_seconds/3600


# In[16]:


trip_length_hours


# In[17]:


trip_length_hours.max()


# In[36]:


ones = np.array([[1,1,1], [1,1,1]])


# In[37]:


zeros = np.array([0,0,0])
zeros = zeros.reshape(1,3)


# In[38]:


concat = np.concatenate([ones, zeros], axis=0)
concat


# In[42]:


twos = np.array([2,2,2])


# In[43]:


twos = twos.reshape(3,1)


# In[45]:


concat_1 = np.concatenate([concat, twos], axis=1)
concat_1

