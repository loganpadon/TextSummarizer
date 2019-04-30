# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:04:55 2019

@author: johnb
https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50

Run this file to create Doc2Vec model
"""

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from retrieve_article import get_articles

train_filename = "training_articles.txt"
model_filename = "doc2vec_model"

"""
Lemmatizes given string
First tokenizes by words using NLTK, then lemmatizes each word using WordNet
"""
def get_lemmas(doc):
    #import nltk
    #nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(doc)
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word)
        lemmas.append(lemma)
    return lemmas

"""
Simple Bag-of-Words similarity metric
Just lemmatizes documents and gets how many tokens in common
"""
def Jaccard_similarity(doc1, doc2):
    doc1_lemmas = get_lemmas(doc1)
    doc2_lemmas = get_lemmas(doc2)
    
    doc1_set = set(doc1_lemmas)
    doc2_set = set(doc2_lemmas)
    intersect = doc1_set.intersection(doc2_set)
    return len(intersect) / (len(doc1_set) + len(doc2_set) - len(intersect))

"""
Similarity using spaCy module
Not sure how it does what it does
"""
def similarity_spacy(doc1, doc2):
    import spacy
    spacy_en = spacy.load('en_core_web_sm')
    spacy_doc1 = spacy_en(doc1)
    spacy_doc2 = spacy_en(doc2)

    return spacy_doc1.similarity(spacy_doc2)

"""
Cosine similarity using TFIDF
"""
def cos_sim_tfidf(doc1, doc2, text=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not text:
        text = [doc1, doc2]
    tfidf = TfidfVectorizer().fit_transform(text)
    pairwise_similarity = (tfidf * tfidf.T).A
    return pairwise_similarity[0][1]
    
"""
Uses pretrained Doc2Vec model 
Functions below for training model based on corpus of articles
Seems to be garbage...
Maybe needs really large corpus
"""
def cos_sim_doc2vec(doc1, doc2):
    from sklearn.metrics.pairwise import cosine_similarity
    
    vectors = list(get_doc_vectors(doc1, doc2))
    return cosine_similarity(vectors)[0][1]

"""
Gets articles using retrieve articles functionality
Saves preprocessed article text to file
Builds training corpus in text file
"""
def _create_train_file(filename, target):
    with open(filename, "w+", encoding="utf8") as train_file:
        num_articles = 0
        for article in get_articles(2017):
            train_file.write(article["preprocessed"] + "\n")
            num_articles += 1
            if num_articles == target:
                break

"""
Reads in training corpus file and converts each to TaggedDocument
""" 
def _read_training_file(filename):
    from gensim.models.doc2vec import TaggedDocument
    with open(filename, "r", encoding="utf8") as train_file:
        for i, line in enumerate(train_file):
            yield TaggedDocument(line, [i])
    
"""
Trains gensim Doc2Vec model based on training corpus in file
Saves trained model to file
"""
def _train_doc2vec_model(target = 100):
    import os
    from gensim.models.doc2vec import Doc2Vec
    if not os.path.isfile(train_filename):
        _create_train_file(train_filename, target)
    training_corpus = list(_read_training_file(train_filename))
    model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(training_corpus)
    model.train(training_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_filename)

"""
Loads Doc2Vec model and vectorizes input documents based on trained model
""" 
def get_doc_vectors(*docs):
    from gensim.models.doc2vec import Doc2Vec
    import os
    if not os.path.isfile(model_filename):
        _train_doc2vec_model(10000)
    model = Doc2Vec.load(model_filename)
    for doc in docs:
        words = word_tokenize(doc)
        yield model.infer_vector(words)

"""
Trains a Doc2Vec model, performs test
"""
def main():
    with open('abstract_sample.txt','r') as file:
        doc1 = file.readline()
        doc2 = file.readline()
    #similarity = cos_sim_doc2vec(doc1, doc2)
    similarity = cos_sim_tfidf(doc1, doc2)
    print(similarity)

if __name__ == "__main__":
   main()