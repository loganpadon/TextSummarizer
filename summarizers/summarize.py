# -*- coding: utf-8 -*-
"""
Main file for project interface
Runs functions for retrieving articles from API and functions for summarization
Saves results/statistics in log file
"""

from basic_text_summary import summarize as basic_summarize
from similarity_matrix_summarizer import sim_matrix_summarize
from text_rank_summarizer import text_rank_summarize
from tf_idf_summarizer import tf_idf_summarize
from frequency_summarizer import FrequencySummarizer

from text_comparison import cos_sim_tfidf
from retrieve_article import get_articles
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from datetime import datetime

import os

"""
Writes relevant data to file
"""
def log(log_filename, abstract, title, topics, types, word_count, results):

    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))
    
    with open(log_filename, "a+", encoding="utf8") as logfile:
        logfile.write("-"*20 + str(datetime.now()) + "-"*20 + "\n")
        logfile.write("Title: " + title.replace("\n","") + "\n")
        logfile.write("Word count: " + str(word_count) + "\n")
        """
        for name, similarity in similarities.items():
            logfile.write(name + ": " + str(similarity) + "\n")
        """
        logfile.write("Abstract: " + abstract.replace("\n","") + "\n")
        for name, result in results.items():
            logfile.write(name + " similarity: " + str(result["similarity"]) + "\n")
            logfile.write(name + " summary: " + result["summary"].replace("\n", "") + "\n")
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
    log_filepath = os.path.join("logs", "summary_log_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt")
    for article in get_articles():
        if articles is None:
            continue
        text = article["preprocessed"]
        title = article["title"].replace("\n", "").replace("\r","")
        abstract = article["description"].replace("\n","").replace(r"\ud"," ").replace("\r","")
        abstract_len = len(sent_tokenize(abstract))
        results = {}

        # Basic summary
        basic_summary = basic_summarize(text, abstract_len)
        # If summary is empty, don't include it
        # Print to see what went wrong
        # When this happens, usually the article was invalid in the first place
        if not basic_summary:
            print("Empty summary for ", title, ":")
            print(text)
            continue
        basic_sim = cos_sim_tfidf(basic_summary, abstract)
        results["basic"] = {"summary" : basic_summary, "similarity" : basic_sim}

        # Similarity matrix summary
        sim_matrix_summary = sim_matrix_summarize(text, abstract_len)
        sim_matrix_sim = cos_sim_tfidf(sim_matrix_summary, abstract)
        results["sim_matrix"] = {"summary": sim_matrix_summary, "similarity" : sim_matrix_sim}

        # Text Rank Summary
        text_rank_summary = text_rank_summarize(text, abstract_len)
        text_rank_sim = cos_sim_tfidf(text_rank_summary, abstract)
        results["text_rank"] = {"summary" : text_rank_summary, "similarity" : text_rank_sim}

        """
        # TFIDF Summary
        tf_idf_summary = tf_idf_summarize(text, title, abstract_len)
        tf_idf_sim = cos_sim_tfidf(tf_idf_summary, abstract)
        results["tfidf"] = {"summary" : tf_idf_summary, "similarity" : tf_idf_sim}
        """

        # Frequency-based summary
        freq_summer = FrequencySummarizer()
        freq_summary = freq_summer.summarize(text, abstract_len)
        freq_sim = cos_sim_tfidf(freq_summary, abstract)
        results["frequency"] = {"summary" : freq_summary, "similarity" : freq_sim}

        # Other data
        word_count = get_word_count(text)
        types = article["subjects"] if "subjects" in article.keys() else []
        topics = article["topics"] if "topics" in article.keys() else []
        #types = article["types"] if "types" in article.keys() else []

        log(log_filepath, abstract, article["title"], topics, types, word_count, results)
        valid_articles += 1
        if valid_articles == articles:
            break

    """
    print()
    print("Articles analyzed: ", valid_articles)
    print("Average for ", name, ":", sum(similarities)/len(similarities))
    print("Min for ", name, ":", min(similarities))
    print("Max for ", name, ":", max(similarities))
    """

def main():
    import sys
    if len(sys.argv) == 2:
        articles = int(sys.argv[1])
        run_summarize(articles)
    else:
        run_summarize()

if __name__ == "__main__":
   main()