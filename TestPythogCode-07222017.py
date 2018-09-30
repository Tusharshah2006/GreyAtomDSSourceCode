
# coding: utf-8

# In[1]:


import os
os.chdir("D:\GreyAtom\Datasets")


# In[2]:


import pandas as pd
wnba = pd.read_csv("wnba.csv")


# In[3]:


wnba.head()


# In[6]:


d = {}
d["Name"] = "Tushar"
d["City"] = "Mumbai"


# In[9]:


d["weather"] = "Rainy"
d


# In[15]:


#drop the column ffrom the dataset
wnba = wnba.drop(["Name"], axis = 1)


# In[12]:


import fuzzywuzzy


# In[13]:


import gTTS


# In[17]:


wnba.shape
wnba


# In[18]:


ls


# In[20]:


wnba["Experience"].value_counts()


# In[23]:


wnba["exp_ordinal"] = wnba["Experience"]
wnba.shape


# In[26]:


wnba.exp_ordinal = pd.to_numeric(wnba.exp_ordinal, errors='coerce').fillna(0).astype(np.int64)


# In[27]:


wnba["exp_ordinal"]


# In[39]:


wnba["exp_ordinal"] = wnba["Experience"].apply(pd.to_numeric, errors = "coerce").fillna(0)


# In[29]:


wnba["exp_ordinal"]


# In[50]:


def exp_ordinal_map(row):
    if row['exp_ordinal'] == 0:
        return "Rookie"
    if (1 < row['exp_ordinal'] <= 3):
        return "Little Experienced"
    if (3 < row["exp_ordinal"] <= 4):
        return "Experienced"
    if (5 < row["exp_ordinal"] <= 10):
        return "Very Experienced"
    else:
        return "Veteran"

wnba["exp_ordinal"] = wnba["Experience"].apply(pd.to_numeric, errors = "coerce").fillna(0)
wnba['exp_ordinal'] = wnba.apply(exp_ordinal_map, axis = 1)
#wnba['exp_ordinal'].value_counts().plot.bar(rot = 45) 
wnba['exp_ordinal'].value_counts().plot.barh() 


# In[53]:


wnba['Pos'].value_counts().plot.barh(title = 'Number of players in WNBA by position')


# In[56]:


wnba['Pos'].value_counts().plot.pie(title = 'Number of players in WNBA by position', figsize = (6, 6), autopct = '%.1f%%')


# In[62]:


wnba['PTS'].plot.hist(bins = 50)


# In[63]:


import os
os.chdir("D:\GreyAtom\Datasets")


# In[66]:


starwars = pd.read_csv("StarWars.csv", encoding="ISO-8859-1")
starwars


# In[67]:


starwars.shape


# In[68]:


list(starwars)


# In[69]:


starwars['RespondentID'].describe()


# In[70]:


starwars['RespondentID'].value_counts()


# In[71]:


starwars['RespondentID'].isnull().value_counts()


# In[76]:


starwars.columns


# In[82]:


starwars['new_column'] = starwars['Do you consider yourself to be a fan of the Star Wars film franchise?'].map({'Yes': True, 'No':False})
starwars['new_column']


# In[84]:


yes_no = {'Yes': True, 'No':False}
for x in ["Have you seen any of the 6 films in the Star Wars franchise?",
         "Do you consider yourself to be a fan of the Star Wars film franchise?"]:
    starwars[x] = starwars[x].map(yes_no)


# In[85]:


starwars.head()


# In[86]:


starwars['RespondentID'].isnull()


# In[152]:


starwars = pd.read_csv("StarWars.csv", encoding="ISO-8859-1")
for x in starwars.iloc[:, 3:9]:
    starwars[x] = starwars[x].notnull()


# In[153]:


starwars


# In[154]:


starwars = starwars.rename(columns={'Which of the following Star Wars films have you seen? Please select all that apply.':'seen_1',
                         'Unnamed: 4':'seen_2','Unnamed: 5':'seen_3','Unnamed: 6':'seen_4','Unnamed: 7':'seen_5','Unnamed: 8':'seen_6'})


# In[126]:


starwars


# In[158]:


starwars = starwars.rename(columns={'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.':'rank_1',
                                    'Unnamed: 10':'rank_2',
                                    'Unnamed: 11':'rank_3',
                                    'Unnamed: 12':'rank_4',
                                    'Unnamed: 13':'rank_5',
                                    'Unnamed: 14':'rank_6'})


# In[159]:


starwars.head()


# In[170]:


#for x in starwars.iloc[1:,9:15]:
#    starwars[x][1:] = starwars[x][1:].astype(float)

#starwars["rank_2"][1]


# In[171]:


starwars.iloc[1:,9:15] = starwars.iloc[1:,9:15].astype(float)
starwars


# In[177]:


res=starwars.iloc[1:,9:15].mean()


# In[179]:


import matplotlib.pyplot as plt
plt.bar(range(len(res)), res)

