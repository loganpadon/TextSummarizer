# -*- coding: utf-8 -*-
"""
Main file for project interface
Runs functions for retrieving articles from API and functions for summarization
Saves results/statistics in log file
"""

from basic_text_summary import summarize as basic_summarize
import text_comparison
from retrieve_article import get_articles
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from datetime import datetime

"""
Writes relevant data to file
"""
def log(log_filename, summary, abstract, title, topics, types, word_count, similarities):
    
    with open(log_filename, "a+", encoding="utf8") as logfile:
        logfile.write("-"*20 + str(datetime.now()) + "-"*20 + "\n")
        logfile.write("Title: " + title.replace("\n","") + "\n")
        logfile.write("Word count: " + str(word_count) + "\n")
        """
        for name, similarity in similarities.items():
            logfile.write(name + ": " + str(similarity) + "\n")
        """
        logfile.write("Similarity: " + str(similarities["TFIDF Similarity"]) + "\n")
        logfile.write("Abstract: " + abstract.replace("\n","") + "\n")
        logfile.write("Basic summary: " + summary.replace("\n","")  +"\n")
        logfile.write("Topics: " + ", ".join(topics) + "\n")
        logfile.write("Types: " + ", ".join(types) + "\n")    

"""
Counts word that match given regex
Currently alphanumeric characters, underscore, dashes
Tokenizes with NLTK 
"""    
def get_word_count(document):
    import re
    count = 0
    for word in word_tokenize(document):
        if re.match("^[A-Za-z0-9_-]*$", word):
            count += 1
    return count

"""
Uses retrieve article functionality, runs all summarization functions
Compares each summary with abstract, logs similarity metric
"""
def run_summarize(articles = 200):
    valid_articles = 0
    all_similarities = {}
    sims_spacy = []
    sims_doc2vec = []
    sims_jaccard = []
    sims_tfidf = []
    log_filename = "summary_log_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    for article in get_articles():
        if articles is None:
            continue
        text = article["preprocessed"]
        title = article["title"].replace("\n", "").replace("\r","")
        abstract = article["description"].replace("\n","").replace(r"\ud"," ").replace("\r","")
        # Summary must be same number of sentences as abstract
        abstract_len = len(sent_tokenize(abstract))
        basic_summary = basic_summarize(text, abstract_len)
        # If summary is empty, don't include it
        # Print to see what went wrong
        if not basic_summary:
            print("Empty summary for ", title, ":")
            print(text)
            continue
        #sim_spacy = text_comparison.similarity_spacy(basic_summary, abstract)
        #sim_doc2vec = text_comparison.cos_sim_doc2vec(basic_summary, abstract)
        #sim_jaccard = text_comparison.Jaccard_similarity(basic_summary, abstract)
        sim_tfidf = text_comparison.cos_sim_tfidf(basic_summary, abstract)
        #sims_spacy.append(sim_spacy)
        #sims_doc2vec.append(sim_doc2vec)
        #sims_jaccard.append(sim_jaccard)
        sims_tfidf.append(sim_tfidf)
        #similarities = {"Spacy Similarity" : sim_spacy, "Doc2Vec Similarity" : sim_doc2vec, 
            #"Jaccard Similarity" : sim_jaccard, "TFIDF Similarity" : sim_tfidf}
        similarities = {"TFIDF Similarity" : sim_tfidf}
        word_count = get_word_count(text)
        types = article["subjects"] if "subjects" in article.keys() else []
        topics = article["topics"] if "topics" in article.keys() else []
        #types = article["types"] if "types" in article.keys() else []
        log(log_filename, basic_summary, abstract, article["title"], 
            topics, types, word_count, similarities)
        valid_articles += 1
        if valid_articles == articles:
            break
    
    #all_similarities = {"Spacy Similarities" : sims_spacy, "Doc2Vec Similarities" : sims_doc2vec,
        #"Jaccard Similarities" : sims_jaccard, "TFIDF Similarities" : sims_tfidf}
    all_similarities = {"TFIDF Similarities" : sims_tfidf}

    print()
    print("Articles analyzed: ", valid_articles)
    for name, similarities in all_similarities.items():
        print("Average for ", name, ":", sum(similarities)/len(similarities))
        print("Min for ", name, ":", min(similarities))
        print("Max for ", name, ":", max(similarities))

def main():
    import sys
    if len(sys.argv) == 2:
        articles = int(sys.argv[1])
        run_summarize(articles)
    else:
        run_summarize()

if __name__ == "__main__":
   main()