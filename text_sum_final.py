#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy



# In[2]:


import en_coref_md

nlp = en_coref_md.load()


# In[3]:


import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt') # one time execution
import re



import io
uploaded=open(r"summarytest.csv")
df = pd.read_csv(r"summarytest.csv",encoding='utf-8')


# In[6]:


df.head()


# In[7]:


# split the the text in the articles into sentences
sentences = []
for s in df['Summary']:
  doc=nlp(s)
  sentences.append(sent_tokenize(s))  


# In[8]:


# flatten the list
sentences = [y for x in sentences for y in x]


# In[9]:


# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]


# In[10]:


#doc=nlp(clean_sentences)


# In[11]:


doc._.has_coref
doc._.coref_clusters
doc._.coref_resolved


# In[12]:


nltk.download('stopwords')# one time execution


# In[13]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[14]:


# function to remove stopwords
def remove_stopwords(sen):
  sen_new = " ".join([i for i in sen if i not in stop_words])
  return sen_new


# In[15]:


# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


# In[16]:


# download pretrained GloVe word embeddings
#! wget http://nlp.stanford.edu/data/glove.6B.zip


# In[17]:


#! unzip glove*.zip


# In[18]:


# Extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()


# In[19]:


sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)


# In[20]:


len(sentence_vectors)


# In[21]:


# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])


# In[22]:


from sklearn.metrics.pairwise import cosine_similarity


# In[23]:


for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


# In[24]:


import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)


# In[25]:


ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)


# In[26]:


# Specify number of sentences to form the summary
sn = 10

# Generate summary
for i in range(sn):
  print(ranked_sentences[i][1])


# Find the original article here https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/
