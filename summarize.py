# -*- coding: utf-8 -*-
"""
Main file for project interface
Runs functions for retrieving articles from API and functions for summarization
Saves results/statistics in log file
"""

from basic_text_summary import summarize as basic_summarize
from text_comparison import cos_sim_doc2vec
from retrieve_article import get_articles
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from datetime import datetime

"""
Writes relevant data to file
"""
def log(log_filename, summary, abstract, title, word_count, similarity):
    
    with open(log_filename, "a+", encoding="utf8") as logfile:
        logfile.write("-"*20 + str(datetime.now()) + "-"*20 + "\n")
        logfile.write("Title: " + title + "\n")
        logfile.write("Word count: " + str(word_count) + "\n")
        logfile.write("Similarity: " + str(similarity) + "\n")
        logfile.write("Abstract: " + abstract + "\n")
        logfile.write("Basic summary: " + summary  +"\n")

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
    similarities = []
    log_filename = "summary_log_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    for article in get_articles():
        if articles is None:
            continue
        text = article["preprocessed"]
        title = article["title"].replace("\n", "")
        abstract = article["description"].replace("\n", "").replace(r"\ud", " ")
        # Summary must be same number of sentences as abstract
        abstract_len = len(sent_tokenize(abstract))
        basic_summary = basic_summarize(text, abstract_len)
        # If summary is empty, don't include it
        # Print to see what went wrong
        if not basic_summary:
            print("Empty summary for ", title, ":")
            print(text)
            continue
        similarity = cos_sim_doc2vec(basic_summary, abstract)
        similarities.append(similarity)
        word_count = get_word_count(text)
        log(log_filename, basic_summary, abstract, article["title"], word_count, similarity)
        valid_articles += 1
        if valid_articles == articles:
            break
    
    print()
    print("Articles analyzed: ", valid_articles)
    print("Average similarity: ", sum(similarities)/len(similarities))
    print("Min similarity: ", min(similarities))
    print("Max similarity: ", max(similarities))

def main():
    import sys
    if len(sys.argv) == 2:
        articles = int(sys.argv[1])
        run_summarize(articles)
    else:
        run_summarize()

if __name__ == "__main__":
   main()