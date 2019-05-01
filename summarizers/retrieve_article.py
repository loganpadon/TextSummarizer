# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:44:27 2019

@author: johnb
"""

import requests
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from math import ceil
from langdetect import detect

excluded_languages = ["uk", "ru"]

def build_url(page = 1, year=2018):
    base = "https://core.ac.uk:443/api-v2/articles/search/"
    api_key = "1cEI2Xf4yzNwiOmQaVDTxlnsuZJdjKro"
    query = "year:" + str(year)
    
    url = base+query+"?pageSize=100&fulltext=true&apiKey="+api_key+"&page="+str(page)
    
    return url

def get_total_pages(year=2018):
    url = build_url(year)
    with requests.get(url) as req:
        response = req.json()
        if not response["status"] == "OK":
            print("Error: " + response["status"])
            return 0
        hits = response["totalHits"]
        pages = ceil(hits / 100)
        
        return pages
    return 0
        
def valid_article(article):
    if not ("description" in article.keys() and 
            "fullText" in article.keys()):
        #print("Does not have abstract/full text")
        return False
    text = article["fullText"].replace("\n","").replace("\t","")
    words = word_tokenize(text)
    # Either article is too short or too long
    # Article that is too long could be bad tokenization
    if len(words) > 5000 or len(words) < 200:
        #print("Invalid word range")
        return False
    #print(detect(text))
    # Only allow English for now
    if not detect(text) == "en":
        #print("Article not in English: ", detect(text))
        return False
    return True
        
"""
Gets rid of certain sentences that are unique to academic papers
Examples:
    links
    information about article upload
    the actual title of the article
TODO: Filter out references
"""
def preprocess(text, title):
    # Find if title is in text
    # Title might not be exact string match
    import re
    processed = ""
    sentences = sent_tokenize(text)
    #print(title)
    #print(sentences)
    for sentence in sentences:
        # Information about upload - don't include
        if "Date originally posted:" in sentence:
            continue
        # Discard sentences with links
        if "https://" in sentence or "http://" in sentence:
            continue
        
        title_found = False
        text_sent_no_space = re.sub(r"\s+", "", sentence.lower())
        for title_sent in sent_tokenize(title):   
            title_sent_no_space = re.sub(r"\s+", "", title_sent.lower())
            if title_sent_no_space in text_sent_no_space:
                title_found = True
        if title_found:
            #print("Title found")
            continue
        
        processed += sentence + " "
    return processed

"""
Retrieves the articles from Core API
Runs filter and preprocessing functions
"""
def get_articles(year=2018):
    import sys
    valid_articles = 0
    all_articles = 0
    max_pages = 10000
    print("Contacting API...")
    pages = get_total_pages(year)
    if pages <= 0:
        return
    print("Starting article retrieval...")
    for page in range(1, min(pages + 1, max_pages + 1)):
        url = build_url(page)
        with requests.get(url) as req:
            response = req.json()
            for article in response["data"]:
                all_articles += 1
                if not valid_article(article):
                    continue
                
                title = article["title"].replace("\n", " ")
                fulltext = article["fullText"].replace("\n"," ").replace("\t"," ")
                text = preprocess(fulltext, title)
                # If preprocessing filters out all text for some reason, 
                # don't bother trying to summarize
                if not text.strip():
                    continue
                
                # Finally, article is valid 
                article["preprocessed"] = text
                
                yield article
                valid_articles += 1
                print("Retrieving articles: ", valid_articles, end="\r")
                sys.stdout.write("\033") # Cursor up one line
    
def main():
    get_articles()
    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()