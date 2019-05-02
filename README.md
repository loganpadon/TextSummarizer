# TextSummarizer
Text summarizer for NLP.
Implements several different text summarization methods and performs analysis on the results.
Uses [CORE API](https://core.ac.uk/services/api/) for input data. 

### External requirements:
* For the neural model, 
[keras-text-summarization](https://github.com/chen0040/keras-text-summarization) 
is required.
* For the TextRank summarization method, 
[gloVe](http://nlp.stanford.edu/data/glove.6B.zip) 
embeddings are required. 
* Many different Python modules are used, including:
  * nltk
  * numpy
  * tensorflow
  * keras
  * matplotlib
  * sklearn
  * pandas
  * statistics
  * networkx
  * langdetect
* There are a few that aren't necessary to run the project but were also used:
  * spacy
  * gensim

### To run the project
Once all the requirements are satisified, all it takes to run the project is `python summarize.py` in the command line from the summarizers folder. The glove file is also required in the summarizers folder. Log files are stored in the logs folder. To analyze, run the command `python make_graphs.py` on the relevant log file. 

### make_graphs.py
* Performs analysis on a log file
* Takes file name as command line argument
* Plots relevant analysis

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

### frequency_summarizer.py:
* Implements Frequency-based summarization
* Contains class FrequencySummarizer, which contains method summarize

### neural_model_prebuilt.py:
* Based off keras-text-summarization module, which implements abstract text summarization using deep learning
* Contains code to train/test a neural model using articles from CORE API
* Exposes method neural_summarize, which uses existing neural model to summarize text

### similarity_matrix_summarizer.py:
* Implements Similarity Matrix summarization
* Method sim_matrix_summarize is used to summarize text

### text_rank_summarizer.py:
* Implements TextRank algorithm for text summarization, similar to PageRank algorithm
* Uses gloVe embeddings for word vectors (glove.6B.100d.txt)
* Method text_rank_summarize is used to summarize text

### tf_idf_summarizer.py:
* Implements TF-IDF Summarization
* Ranks sentences based on multiple factors and chooses the top-scored
* Method tf_idf_summarize is used to summarize text