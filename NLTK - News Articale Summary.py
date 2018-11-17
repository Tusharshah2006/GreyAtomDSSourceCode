
# coding: utf-8

# In[58]:


import urllib.request
import imblearn
import nltk
import operator
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords
from string import punctuation
custom = set(stopwords.words('english')+list(punctuation)+['’', '“', '”'])


# In[42]:


link = "https://qz.com/india/1367236/kerala-floods-show-how-climate-change-threatens-india/"


# In[43]:


req = urllib.request.Request(link)
response = urllib.request.urlopen(req)
the_page = response.read()


# In[44]:


soup = BeautifulSoup(the_page, 'lxml')


# In[45]:


paras = ' '.join([p.text for p in soup.findAll('p')])
print(paras)


# In[46]:


text = paras[:-390]
print(text)


# In[47]:


sentences = sent_tokenize(text.lower())
words = word_tokenize(text.lower())


# In[48]:


print(len(sentences))
print(len(words))
print(len(set(words)))


# In[49]:


words = [w for w in words if w not in custom]


# In[55]:


dictonary = Counter(words)
counts = dictonary.most_common()


# In[57]:


final_words_we_need = [x for x in counts if x[1]>3]
final_words_we_need


# In[60]:


d = {}
for s in sentences:
    score = 0
    for w in final_words_we_need:
        if w[0] in s:
            score = score + w[1]
    d[s] = score
sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_d)


# In[63]:


summary = []
for x in sorted_d[:6]:
    summary.append(x[0])
summary = ''.join(summary[:])
print(summary)

