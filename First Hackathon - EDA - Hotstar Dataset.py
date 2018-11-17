
# coding: utf-8

# In[8]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


os.chdir("C:/Users/chakr/Desktop/GA_Datasets")


# In[10]:


df= pd.read_json("train_data.json")


# In[11]:


#Extrapolating features from the "Time of Day" column
df=df.T


# In[12]:


df["times_only"]=df["tod"].apply(lambda x: [i.rsplit(":")[0] for i in x.rsplit(",")])
df["Morning_Views"]= df["times_only"].apply(lambda x:sum([1 for i in x if int(i)<12]))
df["Afternoon_Views"]= df["times_only"].apply(lambda x:sum([1 for i in x if int(i)>=12 and int(i)<17]))
df["Evening_Views"]= df["times_only"].apply(lambda x:sum([1 for i in x if int(i)>=17 and int(i)<20]))
df["Night_Views"]=df["times_only"].apply(lambda x:sum([1 for i in x if int(i)>=20 and int(i)<=24]))
df["Total_Views"]= df["times_only"].apply(lambda x:len(x))


# In[13]:


df.head()


# In[14]:


#Take the mean of the total views for the positive class and negative class
mean_segment= df.groupby( [ "segment"]).mean()
plt.bar(mean_segment.index,(mean_segment["Total_Views"]))


# In[15]:


#Insight: The positive class seems to have a slightly higher number of total views than the negative class


# In[48]:


df.boxplot(column=['Morning_Views'])


# In[47]:


df.boxplot(column=['Afternoon_Views'])


# In[49]:


df.boxplot(column=['Evening_Views'])


# In[50]:


df.boxplot(column=['Night_Views'])


# In[20]:


#Insight: There seems to be more variation between the morning and afternoon viewership of the positive and negative class
# The evening and night class seems to have roughly equal viewership between the positive and the negative class
#From this it would seem that morning and afternoon classes are better class predictors than evening and night
#Outcome of this exercise: 3 features selected- morning viewers, afternoon viewers and total viewers


# In[21]:


#Feature engineering with cities

a=[]
for i in df['cities']:
    for j in (i.split(',')):
              a.append((j.split(':')[0])) 
            
a=set(a)
metro= ["mumbai","gurgaon","kolkata","delhi","chennai","bangalore","pune","hyderabad","new delhi","delhi","navi mumbai"]
Metros = []
for i in a:
    if i in metro:
        Metros.append(i)
Metros = set(Metros)

Non_Metros = a-Metros
Metros = list(Metros)


# In[22]:


df["city_only"] = df["cities"].apply(lambda x: [i.rsplit(":")[0] for i in x.rsplit(",")])
df["metro"] = df["city_only"].apply(lambda x: sum([1 for i in x if i in Metros]))
df['Non_metro'] = df['city_only'].apply(lambda x: sum([1 for i in x if i not in Metros]))


# In[23]:


mean_segment=df.groupby( [ "segment"]).mean()
mean_segment
plt.bar(mean_segment.index,(mean_segment["metro"]))


# In[24]:


mean_segment=df.groupby( [ "segment"]).mean()
mean_segment
plt.bar(mean_segment.index,(mean_segment["Non_metro"]))


# In[25]:


#Interestingly, the non metro category seems to be a stronger class predictor than the metro category.
#The negative class has a lower mean of no. metro cities that a user watches in, as compared to the positive class


# In[26]:


df["Number_of_cities"]= df["city_only"].apply(lambda x:len(x))


# In[27]:


mean_segment=df.groupby( [ "segment"]).mean()
mean_segment
plt.bar(mean_segment.index,(mean_segment["Number_of_cities"]))


# In[28]:


#Intuition here is that users that watch Hotstar shows in a greater number of cities may be more responsive to advertisements
#However not much difference between the classes here while taking the mean


# In[29]:


#However, what happens if we look at users that watch in more than 5 or 10 cities? 
above_5_metro=df.loc[df["metro"]>5, ["segment"]]
above_5_metro["segment"].value_counts(normalize=True)


# In[30]:


above_10_metro=df.loc[df["metro"]>=10, ["segment"]]
above_10_metro["segment"].value_counts(normalize=True)


# In[31]:


#Intuition above is that at higher levels, a high number of cities per user can be an important class predictor


# In[32]:


#Total time viewed as an indicator of likelihood of responding to advertisement
df["viewing_time_only"]=df["tod"].apply(lambda x: [i.split(":")[1] for i in x.split(",")])
df["Summed_time"]=df["viewing_time_only"].apply(lambda x: sum([int(i) for i in x ]))


# In[33]:


mean_segment= df.groupby( [ "segment"]).mean()
mean_segment
plt.bar(mean_segment.index,(mean_segment["Summed_time"]))
plt.show()


# In[34]:


#Total viewing time seems to be an important class predictor- with the positive class having users with higher viewing times
#In other words the positive class have a higher mean watch time
#people who watch for more time are highly represented in positive class


# In[35]:


df.head()


# In[36]:


q1= df.Summed_time.quantile(.25)
q3= df.Summed_time.quantile(.75)
iqr= q3-q1
# df['Summed_time']>df['Summed_time'].quantile(0.75)
df.describe()


# In[37]:


#Outlier analysis using the "Summed_time" column
df['segment'].loc[df['Summed_time']>(df['Summed_time'].quantile(0.75)-df['Summed_time'].quantile(0.25))*1.5+df['Summed_time'].quantile(0.75)].value_counts(normalize=True)


# In[38]:


#Total dataset split
df.segment.value_counts(normalize=True)


# In[39]:


outlier_subset=df['segment'].loc[df['Summed_time']>(df['Summed_time'].quantile(0.75)-df['Summed_time'].quantile(0.25))*1.5+df['Summed_time'].quantile(0.75)].value_counts(normalize=True)
whole_ds= df.segment.value_counts(normalize=True)


# In[40]:


plt.bar(whole_ds.index,whole_ds)


# In[41]:


plt.bar(outlier_subset.index,outlier_subset)


# In[42]:


df.head()


# In[43]:


get_ipython().set_next_input('What about people who watch more titles');get_ipython().run_line_magic('pinfo', 'titles')
Analyzing titles
df["titles"]=df["titles"].str.split(",")
df["titles_count"]=df["titles"].apply(lambda x:len(x))


# In[ ]:


#Are people who watch more shows/movies highly represented in the positive class?
mean_segment= df.groupby( [ "segment"]).mean()
mean_segment
plt.bar(mean_segment.index,(mean_segment["titles_count"]))
plt.show()


# In[ ]:


#Interestingly, people with a higher mean count of titles watched have a higher representation in the negative class.


# In[ ]:


#Is there a relationship between the titles watched and the amount of time spent on Hotstar?
plt.scatter(df.titles_count,df.Summed_time)
plt.show()


# In[ ]:


mean_segment["Afternoon_Views"]


# In[ ]:


total_time_pos= df.loc[df["segment"]=="pos",["Summed_time"]]
total_time_neg= df.loc[df["segment"]=="neg",["Summed_time"]]


# In[ ]:


total_time_pos.boxplot("Summed_time")
figure= plt.figure()
figure.suptitle("Total Time Viewed")
ax= figure.add_subplot(111)
ax.boxplot(total_time_pos)

ax= figure.add_subplot(121)
ax.boxplot(total_time_neg)
plt.show()


# In[ ]:


total_time_neg.boxplot("Summed_time")


# In[ ]:


total_time_pos= df.loc[df["segment"]=="pos",["Morning_Views"]]
morning_neg= df.loc[df["segment"]=="neg",["Morning_Views"]]


# In[ ]:


total_time_pos= df.loc[df["segment"]=="pos",["Afternoon_Views"]]
total_time_pos= df.loc[df["segment"]=="neg",["Afternoon_Views"]]


# In[53]:


df.boxplot(column=['Summed_time'], figsize=(6,6))


# In[55]:


df["titles"]=df["titles"].str.split(",")
df["titles_count"]=df["titles"].apply(lambda x:len(x))


# In[56]:


df.boxplot(column=['titles_count'], figsize=(6,6))


# In[61]:


df['segment'].loc[df['titles_count']>(df['titles_count'].quantile(0.75)-df['titles_count'].quantile(0.25))*1.5+df['titles_count'].quantile(0.75)].value_counts(normalize=True)


# In[63]:


df.loc[df['titles_count']>(df['titles_count'].quantile(0.75)-df['titles_count'].quantile(0.25))*1.5+df['titles_count'].quantile(0.75)]


# In[64]:


16866/200000


# In[65]:


plt.scatter(df.Morning_Views,df.Summed_time)

