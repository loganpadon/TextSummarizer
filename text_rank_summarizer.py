#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip glove*.zip

import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize


def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def cleans_sentences(sentences):
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    return clean_sentences

# Extract
def word_vectors_glove():
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    len(word_embeddings)
    return word_embeddings

print(len(word_vectors_glove()))

def vectors(clean_sentences):
    word_embeddings = word_vectors_glove()
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    return sentence_vectors, sim_mat

#page rank
import networkx as nx

def ranked_sentences(sentences):
    clean_sentences = cleans_sentences(sentences)
    sentence_vectors, sim_mat = vectors(clean_sentences)
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    
    #Please chang the number of sentences for summary length!!!!!
    # Extract top 10 sentences as the summary
    for i in range(10):
        print(ranked_sentences[i][1])






#comment line 80 though 92 since you will be using you own sentences.
#--------------------------------------------------------------------------------
#change the code below for your sentences
#--------------------------------------------------------------------------------
df = pd.read_csv("tennis_articles_v4.csv")


sentences = []
for s in df['article_text']:
    sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x] # flatten list of sentences:
#--------------------------------------------------------------------------------






#!!!!!!!!!!!!!!!!!!!!!
#run the program
ranked_sentences(sentences)

