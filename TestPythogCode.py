
# coding: utf-8

# In[1]:


2+3


# In[2]:


3**2


# In[3]:


print ("This is first oythod statement")


# In[4]:


print ("4**64")


# In[5]:


print (4**64
      )


# In[10]:


name="Tushar"
age = 30.5


# In[7]:


Type(name)


# In[8]:


type(name)


# In[11]:


type(age)


# In[13]:


None


# Name = None

# In[14]:


Name = None


# In[15]:


Name="ABC"


# In[16]:


print(type(Name))


# In[17]:


s="GreyAtom"


# In[18]:


s[0]


# In[21]:


sentence_1 = "Thsi is a rainy day"


# In[20]:


sentence_1.split(" ")


# In[24]:


l = sentence_1.split(" ")


# In[25]:


l


# In[26]:


sentence_2 = ' '.join(l)


# In[27]:


sentence_2


# In[28]:


name = input("What's your name?")


# name

# In[29]:


name


# In[31]:


name = input ("Your Name?")
age = input ("Age?")
Location = input ("City?")

print ("My Name is %s, I am %s year old andc I live in %s" % (name,age,Location))


# In[32]:


bit.ly/7julydsmp


# In[33]:


import os
os.chdir("D:\GreyAtom\PythonSourceCode")


# In[34]:


import math


# In[35]:


print (math.sqrt(3))


# In[36]:


name="test test test"


# In[39]:


name.title()


# In[40]:


a = {}


# In[41]:


type(a)


# In[53]:


a = int(input ("Enter Number 1"))
b = int(input ("Enter Number 2"))
c = a*b
print(c)
print ("The product of a and b is %s" %c)


# In[60]:


aDict = {}
aDict["Name"] = "Tushar"
aDict["Age"] = "29"
aDict["Location"] = "Mumbai"
aDict["Year"] = 2018

for aKey in aDict.items():
    print(aKey[0], aKey[1])
    


# In[63]:


import math as mp
a = int(input("Enter the number to find the square root "))
print(mp.sqrt(a))


# In[68]:


lst = []
for x in range(100):
    if (x%3 == 0):
        lst.append(x)

print (lst)


# In[72]:


cities = ['City1', 'City2', 'City3']
weathers = ['Rain', 'Sun', 'Fog']

student1 = {}

student1 = dict(zip(cities, weathers))

print (student1)


# In[74]:


result = [x for x in range(100) if x%5==0]
print (result)


# In[77]:


import os
os.getcwd()


# In[78]:


os.chdir("D:\GreyAtom\input")


# In[79]:


os.getcwd()


# In[80]:


ls


# In[81]:


dir


# In[86]:


dir()


# In[83]:


ls


# In[88]:


with open('sample.txt', 'r') as f:
    for line in f:
        print(line)


# In[90]:


with open('sample.txt', 'a') as f:
    f.write("This line is appened from python!\n")
    f.write("This is another lineappened from python!\n")


# In[91]:


a = int(input ("Enter Number 1"))
b = int(input ("Enter Number 2"))
c = a*b
print(c)
print ("The product of a and b is %s" %c)


# In[94]:


def multiply(param1, param2):
    return (a*b)

a = int(input ("Enter Number 1"))
b = int(input ("Enter Number 2"))
c = multiply(a, b)
print(c)


# In[95]:


def multiply(param1, param2):
    return (param1*param2)

a = int(input ("Enter Number 1"))
b = int(input ("Enter Number 2"))
c = multiply(a, b)
print(c)

