from flask import Flask, render_template, request
import nltk 
import numpy as np
import spacy
import en_coref_md
from nltk import word_tokenize,sent_tokenize
from gensim.models import Word2Vec
import xml.etree.ElementTree as ET
import pymongo
from pymongo import MongoClient
import pprint
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.corpus import stopwords
client = MongoClient()
db = client.corpustest
collection = db.corpus
nltk.download('punkt') 
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
	if request.method == 'POST':
		result = request.form
		x=result['X']
		y=result['Y']
		
		search = collection.find({"$text": {"$search": x}},{"$text": {"$search": y}}).limit(10)
		print(search)
		print(collection.find())
		




		nlp = en_coref_md.load()	
		uploaded=open(r"C:\Users\ablaz\Desktop\UI\summarytest.csv")
		df = pd.read_csv(r"C:\Users\ablaz\Desktop\UI\summarytest.csv",encoding='utf-8')
		df.head()

		# split the the text in the articles into sentences
		sentences = []
		for s in df['Summary']:
		  doc=nlp(s)
		  sentences.append(sent_tokenize(s))  

		# flatten the list
		sentences = [y for x in sentences for y in x]
		# remove punctuations, numbers and special characters
		clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
		# make alphabets lowercase
		clean_sentences = [s.lower() for s in clean_sentences]
		doc._.has_coref
		doc._.coref_clusters
		doc._.coref_resolved
		nltk.download('stopwords')		
		stop_words = stopwords.words('english')
		# function to remove stopwords
		def remove_stopwords(sen):
		  sen_new = " ".join([i for i in sen if i not in stop_words])
		  return sen_new
		# remove stopwords from the sentences
		clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

		word_embeddings = {}
		f = open('glove_50_300_2.txt', encoding='utf-8')
		for line in f:
		    values = line.split()
		    word = values[0]
		    coefs = np.asarray(values[1:], dtype='float32')
		    word_embeddings[word] = coefs
		f.close()

		sentence_vectors = []
		for i in clean_sentences:
		  if len(i) != 0:
		    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
		  else:
		    v = np.zeros((100,))
		  sentence_vectors.append(v)

		len(sentence_vectors)


		sim_mat = np.zeros([len(sentences), len(sentences)])
		for i in range(len(sentences)):
		  for j in range(len(sentences)):
		    if i != j:
		      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

		nx_graph = nx.from_numpy_array(sim_mat)
		scores = nx.pagerank(nx_graph)
		ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)		
		sn = 10
		for i in range(sn):
		  print(ranked_sentences[i][1])	



		model = Word2Vec.load("word2vec.model")
		sim = model.similarity(x,y)
		print(sim)



				
	return render_template("result.html",result = result)

if __name__ == '__main__':
   app.run(debug = True)


