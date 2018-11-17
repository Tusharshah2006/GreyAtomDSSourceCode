
# coding: utf-8

# In[125]:


import os 
import pandas as pd
import numpy as np


# In[126]:


os.chdir("D:/GreyAtom/Hackathon/hotstar_dataset")
df = pd.read_json("train_data.json")


# In[127]:


dfT = df.T
dfT.head()


# In[128]:


dfT["titles"]=dfT["titles"].str.split(",")
dfT["titles_count"]=dfT["titles"].apply(lambda x:len(x))


# In[129]:


dfT.head()


# In[130]:


Metros = ["mumbai", "gurgaon", "kolkata", "delhi", "chennai", "bangalore", "pune", "hyderabad", "new delhi", "delhi", "navi mumbai"]

dfT["city_only"] = dfT["cities"].apply(lambda x: [i.rsplit(":")[0] for i in x.rsplit(",")])
dfT["metro"] = dfT["city_only"].apply(lambda x: sum([1 for i in x if i in Metros]))
dfT['non_metro'] = dfT['city_only'].apply(lambda x: sum([1 for i in x if i not in Metros]))


# In[131]:


dfT.head()


# In[132]:


dfT["final_hour"] = dfT["tod"].apply(lambda x: [i.rsplit(":")[0] for i in x.rsplit(",")])

dfT["Morning Viewer"] = dfT["final_hour"].apply(lambda x: sum([1 for i in x if int(i)<=12]))
dfT["Afternoon Viewers"] = dfT["final_hour"].apply(lambda x: sum([1 for i in x if int(i)>12 and int(i)<17]))
dfT["Evening Viewers"] = dfT["final_hour"].apply(lambda x:sum([1 for i in x if int(i)>=17 and int(i)<20]))
dfT["Night Viewers"] = dfT["final_hour"].apply(lambda x: sum([1 for i in x if int(i)>=20 and int(i)<=24]))


# In[133]:


dfT.head()


# In[134]:


weekdays = [2, 3, 4, 5, 6]
dfT["dow_list"] = dfT["dow"].apply(lambda x: [i.rsplit(":")[0] for i in x.rsplit(",")])

# dfT["Sunday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 1]))
# dfT["Monday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 2]))
# dfT["Tuesday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 3]))
# dfT["Wednesday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 4]))
# dfT["Thursday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 5]))
# dfT["Friday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 6]))
# dfT["Saturday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) == 7]))

dfT["Weekday"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) in weekdays]))
dfT["Weekend"] = dfT["dow_list"].apply(lambda x: sum([1 for i in x if int(i) not in weekdays]))


# In[135]:


dfT.head()


# In[136]:


Sports = ['Cricket', 'Football', 'Badminton', 'Sport', 'Formula1', 'Hockey', 'Kabaddi', 'Table Tennis', 'Tennis', 'Volleyball', 'Athletics', 'FormulaE', 'Boxing', 'Swimming', 'IndiaVsSa']
Television = ['Wildlife', 'LiveTV', 'TalkShow', 'Reality', 'Awards', 'Travel', 'Science', 'Documentary', 'Mythology', 'Kids']
Movies = ['Drama', 'Family', 'Crime', 'Romance', 'Action', 'Comedy', 'Thriller', 'Teen', 'Horror', 'NA']

dfT["genres_list"] = dfT["genres"].apply(lambda x: [i.rsplit(":")[0] for i in x.rsplit(",")])

# genres_count = {}
# for i in dfT['genres']:
#     for j in (i.split(',')):
#         if j.split(':')[0] not in genres_count:
#              genres_count[j.split(':')[0]]= 1
#         else:
#              genres_count[j.split(':')[0]] += 1

# for key, value in genres_count.items():
#     dfT[key] = ""
#     dfT[key] = dfT["genres_list"].apply(lambda x: sum([1 for i in x if i == key]))

dfT['Sports'] = dfT["genres_list"].apply(lambda x: sum([1 for i in x if i in Sports]))
dfT['Television'] = dfT["genres_list"].apply(lambda x: sum([1 for i in x if i in Television]))
dfT['Movies'] = dfT["genres_list"].apply(lambda x: sum([1 for i in x if i in Movies]))


# In[137]:


dfT.head()


# In[138]:


dfT["viewing_time_only"]=dfT["tod"].apply(lambda x: [i.split(":")[1] for i in x.split(",")])
dfT["Summed_time"]=dfT["viewing_time_only"].apply(lambda x: sum([int(i) for i in x ]))

dfT["Total_Genre_Count"] = dfT["Sports"] + dfT["Television"] + dfT["Movies"]


# In[139]:


dfT.head()


# In[140]:


dfT = dfT.drop(['cities'],1)


# In[141]:


dfT=dfT.drop(['dow','genres','titles','tod','city_only','final_hour','viewing_time_only','dow_list'],1)


# In[142]:


dfT=dfT.drop(['genres_list'],1)


# In[143]:


dfT.head()


# In[144]:


dfT['segment']=dfT['segment'].replace('neg',0).replace('pos',1)


# In[145]:


dfT.head()

