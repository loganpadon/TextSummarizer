# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:04:55 2019

@author: johnb
https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
"""

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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

def Jaccard_similarity(doc1, doc2):
    doc1_lemmas = get_lemmas(doc1)
    doc2_lemmas = get_lemmas(doc2)
    
    doc1_set = set(doc1_lemmas)
    doc2_set = set(doc2_lemmas)
    intersect = doc1_set.intersection(doc2_set)
    return len(intersect) / (len(doc1_set) + len(doc2_set) - len(intersect))

def main():
    doc = ""
    with open('abstract_sample.txt','r') as file:
        doc = file.readline()
    lemmas = get_lemmas(doc)

main()