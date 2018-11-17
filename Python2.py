
# coding: utf-8

# In[2]:


list_1 = []
for x in range(1,11):
    list_1.append(x)
list_1


# In[10]:


list_2 = []
def sequence_a_num(x):
    return(x**2)

for x in list_1:
    list_2.append(sequence_a_num(x))


# In[11]:


list_2


# In[12]:


list_2 = map(sequence_a_num, list_1)


# In[13]:


list(list_2)


# In[14]:


import os
os.chdir("D:\GreyAtom\Datasets")


# In[15]:


os.getcwd()


# In[17]:


import pandas as pd
df = pd.read_csv("ipl_matches_small.csv")


# In[19]:


df.head(2)


# In[24]:


list_1 = [1,2,3,4,5]
list_2 = [6,7,8,9,10]
list_3 = []

for x, y in zip(list_1, list_2):
    list_3.append(x+y)


# In[25]:


print (list_3)


# In[33]:


import math as m
def log10(number):
    return m.log10(number)


# In[40]:


import math
list_1 = [2,5,3,8,5,9]
results = []
for x in list_1:
    results.append(math.log10(x))
print(results)


# In[41]:


import math
list_1 = [2,5,3,8,5,9]
results = list(map(lambda x: math.log10(x), list_1))


# In[39]:


print(results)


# In[44]:


print (sorted(results, reverse=True))


# In[45]:


def li(*args):
    for x in args:
        print (x)

li(3,6,5,6,9)


# In[48]:


def multiply(*args):
    z = 2
    for x in args:
        print (z*x)

multiply(3,6,5,6,9)


# In[49]:


list_1 = [[1,2,3], [4,5,6], [7,8,9]]
import numpy as np
list_beautified = np.array(list_1)


# In[52]:


list_beautified


# In[51]:


list_beautified.shape


# In[53]:


print(type(list_beautified))


# In[56]:


print(list_beautified[1,2])


# In[57]:


#Generating an identity matrix of say size 5
our_identity_matrix = np.eye(5)
our_identity_matrix


# In[58]:


os.chdir("D:\GreyAtom\Datasets")


# In[59]:


os.getDir()


# In[60]:


os.getcwd()


# In[61]:


ls


# In[66]:


import numpy as npn
weather = np.genfromtxt("weather_small_2012.csv", delimiter=",")


# In[67]:


weather


# In[68]:


print(weather[1])


# In[84]:


temperature = (weather[1:,1])
print(type(temperature[1]))


# In[90]:


def CelsiustoFahrenheit(celsius):
    Fahrenheit = []
    for x in celsius:
        Fahrenheit.append((x*9/5)+32)
    print(Fahrenheit)
    
CelsiustoFahrenheit(temperature)


# In[91]:


print(Fahrenheit)


# In[92]:


temperature = (weather[1:,1])
farenheit = (temperature*9/5)+32
farenheit


# In[97]:


farenheit_above_30 = farenheit>30
print(farenheit_above_30)
wind_speed = (weather[1:,4])
print(len(wind_speed[farenheit_above_30]))
print(wind_speed[farenheit_above_30])


# In[102]:


print("Max", temperature.max())
print("Min", temperature.min())
print("Mean Temp", temperature.mean())
print("Argmax", temperature.argmax())
print("Argmin", temperature.argmin())


# In[108]:


arr = np.random.randint(100, size=9).reshape(3,3)


# In[109]:


arr


# In[110]:


import pandas as pd
df = pd.read_csv("ipl_matches_small.csv", header=None)


# In[116]:


df.head()


# In[118]:


df.iloc[:,2]


# In[120]:


df.loc["city"]

