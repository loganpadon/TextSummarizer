# see https://github.com/chen0040/keras-text-summarization
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from keras_text_summarization.library.utility.plot_utils import plot_and_save_history
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
from keras_text_summarization.library.applications.fake_news_loader import fit_text
import numpy as np
from retrieve_article import get_articles


def train():
    LOAD_EXISTING_WEIGHTS = False
    LOAD_DFARTICLES = False

    np.random.seed(42)
    data_dir_path = 'E:\PycharmProjects\TextSummarizer\demo\data'
    report_dir_path = 'E:\PycharmProjects\TextSummarizer\demo\\reports'
    model_dir_path = 'E:\PycharmProjects\TextSummarizer\demo\models'

    print('loading training data')
    if not LOAD_DFARTICLES:
        df = pd.DataFrame(columns=['abstract', 'text'])
        i = 0
        for article in get_articles(year=2017):
            print(i)
            tempDF = pd.DataFrame({'abstract': [article['description']], 'text': [article['fullText']]})
            df = df.append(tempDF, ignore_index=True)
            if i % 10 == 0:
                with open('dfArticles2017.pkl', 'wb') as f:
                    print("dumpin time")
                    pickle.dump([df, i], f)
            # if i >= 100:
            #     break
            i += 1
    else:
        pickle_in = open("dfArticles2017.pkl", "rb")
        asdf = pickle.load(pickle_in)
        df = asdf[0]
        i = asdf[1]

    print('extract configuration from input texts ...')
    Y = df.abstract
    X = df['text']

    # print('loading csv file ...')
    # df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")
    #
    # print('extract configuration from input texts ...')
    # Y = df.title
    # X = df['text']

    config = fit_text(X, Y)

    summarizer = Seq2SeqSummarizer(config)

    if LOAD_EXISTING_WEIGHTS:
        summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    history = summarizer.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=100)

    history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history.png'
    if LOAD_EXISTING_WEIGHTS:
        history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history-v' + str(
            summarizer.version) + '.png'
    plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={'loss', 'acc'})


def test():
    LOAD_DFARTICLES = False
    np.random.seed(42)
    data_dir_path = './data'  # refers to the demo/data folder
    model_dir_path = './models'  # refers to the demo/models folder

    print('loading validation data')
    if not LOAD_DFARTICLES:
        df = pd.DataFrame(columns=['abstract', 'text'])
        i = 0
        for article in get_articles(year=2018):
            print(i)
            tempDF = pd.DataFrame({'abstract': [article['description']], 'text': [article['fullText']]})
            df = df.append(tempDF, ignore_index=True)
            if i % 10 == 0:
                with open('dfArticles2018.pkl', 'wb') as f:
                    print("dumpin time")
                    pickle.dump([df, i], f)
            if i >= 100:
                break
            i += 1
    else:
        pickle_in = open("dfArticles2018.pkl", "rb")
        asdf = pickle.load(pickle_in)
        df = asdf[0]
        i = asdf[1]
    Y = df.abstract
    X = df['text']

    config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path), allow_pickle=True).item()

    summarizer = Seq2SeqSummarizer(config)
    summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    print('start predicting ...')
    for i in range(20):
        x = X[i]
        actual_headline = Y[i]
        headline = summarizer.summarize(x)
        print('Article: ', x)
        print('Generated Headline: ', headline)
        print('Original Headline: ', actual_headline)

def neural_summarize(doc):
    np.random.seed(42)
    data_dir_path = './data'  # refers to the demo/data folder
    model_dir_path = './models'  # refers to the demo/models folder

    config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path), allow_pickle=True).item()

    summarizer = Seq2SeqSummarizer(config)
    summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))
    headline = summarizer.summarize(doc)
    #print('Generated Headline: ', headline)
    return headline



if __name__ == "__main__":
    train()
    test()