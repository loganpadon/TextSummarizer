# -*- coding: utf-8 -*-
"""
Main file for project interface
Runs functions for retrieving articles from API and functions for summarization
Saves results/statistics in log file
"""

from basic_text_summary import summarize
from text_comparison import Jaccard_similarity
from retrieve_article import get_articles
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import statistics

def log(summary, abstract, title, similarity):
    from datetime import date
    from datetime import datetime
    log_filename = "summary_log_" + str(date.today()) + ".txt"
    with open(log_filename, "a+", encoding="utf8") as logfile:
        logfile.write("--------------------" + str(datetime.now()) + "--------------------\n")
        logfile.write("Summarization comparison for article: " + title + "\n")
        logfile.write("Similarity: " + str(similarity) + "\n")
        logfile.write("Abstract: " + abstract + "\n")
        logfile.write("Automatic summary: " + summary  +"\n")

def run_summarize(articles = 200):
    valid_articles = 0
    similarities = []
    for article in get_articles():
        if articles is None:
            continue
        text = article["preprocessed"]
        title = article["title"].replace("\n", "")
        abstract = article["description"].replace("\n", "").replace(r"\ud", " ")
        abstract_len = len(sent_tokenize(abstract))
        summary = summarize(text, abstract_len)
        # If summary is empty, don't include it
        # Print to see what went wrong
        if not summary:
            print("Empty summary for ", title, ":")
            print(text)
            continue
        similarity = Jaccard_similarity(summary, abstract)
        similarities.append(similarity)
        log(summary, abstract, article["title"], similarity)
        valid_articles += 1
        if valid_articles == articles:
            break
    
    print("Articles analyzed: ", valid_articles)
    print("Average similarity: ", statistics.mean(similarities))
    print("Min similarity: ", min(similarities))
    print("Max similarity: ", max(similarities))


def main():
    import sys
    if len(sys.argv) == 2:
        articles = sys.argv[1]
        run_summarize(articles)
    else:
        run_summarize()

main()  