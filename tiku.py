import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt') # one time execution
import re

df = pd.read_csv('summarytest.csv')

df.head()

# split the the text in the articles into sentences
sentences = []
for s in df['Summary']:
  sentences.append(sent_tokenize(s))

# flatten the list
sentences = [y for x in sentences for y in x]

# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]

nltk.download('stopwords')# one time execution

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(sen):
  sen_new = " ".join([i for i in sen if i not in stop_words])
  return sen_new

# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]




def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,encoding="utf-8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        try:
          embedding = np.array([float(val) for val in splitLine[1:]],dtype = "float32")
        except ValueError:
          print("error")
          break
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model
loadGloveModel('glove.840B.300D.txt')

"""
word_embeddings = {}
f = open('glove.840B.300d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float')
    word_embeddings[word] = coefs
f.close()"""

sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)

len(sentence_vectors)

"""The next step is to find similarities among the sentences. We will use cosine similarity to find similarity between a pair of sentences. Let's create an empty similarity matrix for this task and populate it with cosine similarities of the sentences."""

# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])

from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# Specify number of sentences to form the summary
sn = 10

# Generate summary
for i in range(sn):
  print(ranked_sentences[i][1])

