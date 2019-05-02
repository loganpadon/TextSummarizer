# TextSummarizer
Text summarizer for NLP.

External requirements:
* For the neural model, [keras-text-summarization] https://github.com/chen0040/keras-text-summarization is required.
* For the TextRanke summarization method, [gloVe] http://nlp.stanford.edu/data/glove.6B.zip embeddings are required. 

### basic_text_summary.py:
* Code taken from Python extractive text summarization tutorial
* Uses simple word frequency to score sentences
* Very basic, somewhat effective
* Method used: summarize
  * parameters: text, size
  * text: body of text, in one string
  * size: target number of sentences for the summary
* summarizer will pick sentences with top *size* scores

### retrieve_article.py:
* Uses Core API to get scholarly articles from database
* Preprocesses article text
* Filters out articles that are not suitable (e.g. not English, not enough words, no abstract)
* Use get_articles() method for retrieving articles
* Has optional parameter for year, which is 2018 by default
  * Note: get_articles() is generator function, which doesn't stop until it retrieves all articles that match query

### summarizer.py:
* Wrapper program that retrieves articles and uses each summary method on articles
* Logs summaries and statistics for each program run into log file

### text_comparison.py:
* Provides several methods for text similarity measurement
* Jaccard similarity: simple BOW intersection count, normalized
* Doc2Vec: method provided by gensim - creates vectors using documents, instead of word vectors
* spaCy: text similarity implemented by spaCy
* TFIDF: cosine similarity using vectors created by TFIDF

    
