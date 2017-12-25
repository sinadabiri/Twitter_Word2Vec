from scipy.spatial.distance import cosine
from gensim.models.keyedvectors import KeyedVectors
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import pickle

# Create Primary and secondary keywords for analysis
with open('All-TI-Keywords-Weights.csv', 'r', newline='') as handle:
    reader = csv.reader(handle)
    keywords = []
    weights = []
    for row in reader:
        keywords.append(row[0])
        weights.append(float(row[1]))
All_Keywords_Weights = dict(zip(keywords, weights))


class Word2vecClassifier:
    def __init__(self, num_words_mu=40, extract_start=1000, extract_end=2000,
                 training_start=0, training_end=1000, keywords=None):
        self.num_words_mu = num_words_mu
        self.extract_start = extract_start
        self.extract_end = extract_end
        self.training_start = training_start
        self.training_end = training_end

        # The most frequent words in a given training set
        if keywords == None:
            vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
            tweet_corpus = [tweet[2] for tweet in training_tweets[self.extract_start: self.extract_end]
                            if int(tweet[0]) == 1]
            document_term = vectorizer.fit_transform(tweet_corpus).toarray()
            name = vectorizer.get_feature_names()
            frequency = np.sum(document_term, axis=0)
            sort_index = np.argsort(frequency)
            sort_index = sort_index[::-1]
            sort_index = sort_index[:150]
            get_keywords = [name[index] for index in sort_index]
        else:
            get_keywords = keywords

        # Compute mu: the average of word vectors of the most frequent words in traffic dictionary.
        keywords_word2vec = []
        final_keywords = []
        for word in get_keywords:
            try:
                keywords_word2vec.append(Word2Vec_model[word])
                final_keywords.append(word)
            except KeyError:
                pass
        self.keywords = final_keywords[:self.num_words_mu]
        keywords_word2vec = keywords_word2vec[:self.num_words_mu]

        self.mu = np.mean(np.array(keywords_word2vec), axis=0)

    def tweet_similarity_index(self, tweet):
        """
        :param mu: the average of word vectors of the most frequent words in traffic dictionary.
        :param tweet: the subject tweet
        :return: tsi: Tweet similarity index
        """
        vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
        # DocumentTerm_Array = vectorizer.fit_transform(tweet[2]).toarray()
        analyze = vectorizer.build_analyzer()
        all_words_similarity = []
        for word in analyze(tweet[2]):
            try:
                word_vec = Word2Vec_model[word]
                similarity = 1 - cosine(self.mu, word_vec)
                all_words_similarity.append(similarity)
            except KeyError:
                pass
        if all_words_similarity == []:
            tsi = 0
        else:
            tsi = sum(all_words_similarity) / len(all_words_similarity)
        return tsi

    def compute_threshold(self):
        # Calculating the threshold
        traffic_tsi = []
        non_traffic_tsi = []
        for tweet in training_tweets[self.training_start: self.training_end]:
            if int(tweet[0]) == 1:
                traffic_tsi.append(self.tweet_similarity_index(tweet))
            elif int(tweet[0]) == 0:
                non_traffic_tsi.append(self.tweet_similarity_index(tweet))

        threshold = (sum(traffic_tsi) / len(traffic_tsi) + sum(non_traffic_tsi) / len(non_traffic_tsi)) / 2
        return threshold

    def prediction_label(self):
        threshold = self.compute_threshold()
        true_labels = [int(tweet[0]) for tweet in test_tweets]
        pred_labels = []
        for tweet in test_tweets:
            tsi = self.tweet_similarity_index(tweet)
            if tsi >= threshold:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        return true_labels, pred_labels


# Evaluating on test set

filename = '../LSTM-CNN code/GoogleNews-vectors-negative300.bin'
Word2Vec_model = KeyedVectors.load_word2vec_format(filename, binary=True, encoding='latin-1')
#filename = '../LSTM-CNN code/word2vec_twitter_model.bin'
#Word2Vec_model = KeyedVectors.load_word2vec_format(filename, binary=True, encoding='latin-1')

filename = '../TI+NTI Data Collection+Classification/1_TrainingSet_2Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    training_tweets = []
    for tweet in reader:
        training_tweets.append(tweet)
# Parsing test data
filename = '../TI+NTI Data Collection+Classification/1_TestSet_2Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    test_tweets = []
    for tweet in reader:
        test_tweets.append(tweet)

if __name__ == '__main__':
    # Find test accuracy for arbitrary training set
    clf = Word2vecClassifier(training_start=10000, training_end=11000, extract_start=11000,
                             extract_end=12000, num_words_mu=40, keywords=None)
    true_labels, pred_labels = clf.prediction_label()
    print('Accuracy score: ', accuracy_score(true_labels, pred_labels) * 100)
    print(classification_report(true_labels, pred_labels, digits=3))
    print('Confusion Matrix: ', confusion_matrix(true_labels, pred_labels))

    ''''
    # Find accuracy for various set of keywords:
    rand_words = ['milk', 'book', 'mouse', 'tea', 'computer', 'shoes', 'watch', 'pen', 'chair', 'tissue']
    select_words = ['mm', 'ave', 'mile', 'update', 'pa', 'delays', 'wb', 'southbound', 'delay', 'west']
    clf = Word2vecClassifier(training_start=9000, training_end=10000, extract_start=11000,
                                  extract_end=12000, num_words_mu=40, keywords=None)
    true_labels, pred_labels = clf.prediction_label()
    print('Accuracy score: ', accuracy_score(true_labels, pred_labels) * 100)
    print(classification_report(true_labels, pred_labels, digits=3))
    print('Confusion Matrix: ', confusion_matrix(true_labels, pred_labels))
    '''

    '''
    # Find accuracy based on different number of words
    num_words = [1, 2, 3, 4, 5, 6, 7, 8, 9] + list(np.linspace(10, 100, 10, dtype=int))
    plt.figure(1)
    accuracy = []
    for item in num_words:
        clf = Word2vecClassifier(training_start=0, training_end=1000, extract_start=2000,
                                  extract_end=3000, num_words_mu=item)
        # print('Traffic Keywords: ', clf.keywords)
        true_labels, pred_labels = clf.prediction_label()
        # print('Accuracy score: ', accuracy_score(true_labels, pred_labels) * 100)
        accuracy.append(accuracy_score(true_labels, pred_labels) * 100)
    print("Accuracy varying num_words_mu: ", accuracy)
    print('The traffic keywords: ', clf.keywords)

    #Plot accuracy
    plt.plot(num_words, accuracy)
    plt.xticks(np.arange(0, max(num_words)+1, 10))
    plt.yticks(np.arange(91, 101, 1))
    plt.xlabel('Number of traffic keywords')
    plt.ylabel('Test accuracy (%)')
    plt.savefig('Paper2_Accuracy_num_keywords', dpi=600)
    plt.show()
    a = 2
    '''
    '''
    # Accuracy for TR tweets with no keywords
    clf = Word2vecClassifier(training_start=0, training_end=1000, extract_start=1000,
                                  extract_end=2000, num_words_mu=10)
    keywords_10 = clf.keywords[:10]
    vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
    analyze = vectorizer.build_analyzer()
    test_tweets_traffic = [tweet for tweet in test_tweets if int(tweet[0]) == 1]
    tweet_no_keywords = []
    for tweet in test_tweets_traffic:
        words = analyze(tweet[2])
        if all(word not in keywords_10 for word in words):
            tweet_no_keywords.append(tweet)
    test_tweets = tweet_no_keywords
    true_labels, pred_labels = clf.prediction_label()
    print('Accuracy score: ', accuracy_score(true_labels, pred_labels) * 100)
    selected_tweets = [tweet for i, tweet in enumerate(test_tweets) if true_labels[i] == pred_labels[i]]
    with open('Paper2_traffic_tweets_no_keywords.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(keywords_10)
        for tweet in selected_tweets:
            writer.writerow([tweet[0], tweet[1], tweet[2]])
    '''
    '''
    # Qualitative assessment
    clf = Word2vecClassifier(training_start=0, training_end=1000, extract_start=1000,
                              extract_end=2000, num_words_mu=40)
    true_labels, pred_labels = clf.prediction_label()

    wrong_tweets_corpus = []
    for i in range(len(true_labels)):
        wrong_tweet = []
        if true_labels[i] != pred_labels[i]:
            wrong_tweet.append(pred_labels[i])
            wrong_tweet.append(true_labels[i])
            wrong_tweet.append(test_tweets[i][1])
            wrong_tweet.append(test_tweets[i][2])
            wrong_tweets_corpus.append(wrong_tweet)

    with open('Paper2_wrong_classified_tweets.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predict', 'True', 'id', 'Text'])
        for tweet in wrong_tweets_corpus:
            writer.writerow(tweet)
    a = 3
    '''
    # Find the most similar words to the most frequent keywords
    clf = clf = Word2vecClassifier(training_start=0, training_end=1000, extract_start=0,
                                  extract_end=40000, num_words_mu=10)
    traffic_keywords = {'highway': [], 'exit': [], 'lane': [], 'crash': [], 'blocked': [],
                        'cleared': [], 'closed': [], 'updated': [], 'traffic': [], 'vehicle': []}

    for keyword in traffic_keywords:
        similarity = []
        a = Word2Vec_model.index2word
        for word in Word2Vec_model.index2word:
            word_vec = Word2Vec_model[word]
            sim = 1 - cosine(Word2Vec_model[keyword], Word2Vec_model[word])
            similarity.append([word, sim])
        sort_similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
        sort_similarity = sort_similarity[:100]
        [traffic_keywords[keyword].append(item[0]) for item in sort_similarity]

    print('the most similar words to the most frequent keywords ', traffic_keywords)
    a = 2

    '''
    # Find accuracy for various datasets
    acc_keywords = []
    for i in range(39):
        clf = Word2vecClassifier(training_start=1000*i, training_end=1000*(i+1), extract_start=1000*(i+1),
                                  extract_end=1000*(i+2), num_words_mu=40)
        true_labels, pred_labels = clf.prediction_label()
        acc_keywords.append([accuracy_score(true_labels, pred_labels) * 100, clf.keywords])

    with open('Paper2_allsest_accuracy_Google.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        for row in acc_keywords:
            writer.writerow([row[0], row[1]])
    '''
