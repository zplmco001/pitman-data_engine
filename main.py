import time

import nltk
import csv

import schedule

from cnn import SimpleCNN, test_model
from logistic_regression import LogisticRegression
from naive_bayes import summary, calculate

sample = """Dave watched as the forest burned up on the hill,
only a few miles from his house. The car had
been hastily packed and Marta was inside trying to round
up the last of the pets. "Where could she be?" he wondered
as he continued to wait for Marta to appear with the pets."""


def feature_extractor(text, pos_common_list, neg_common_list):
    result = dict()
    pos_word_counter = 0
    neg_word_counter = 0
    word_counter = 0
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in [i[0] for i in pos_common_list]:
                pos_word_counter += 1
            if word.lower() in [i[0] for i in neg_common_list]:
                neg_word_counter += 1
            word_counter += 1

    result['pos_words'] = pos_word_counter
    result['neg_words'] = neg_word_counter
    result['word_count'] = word_counter
    return result


def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted_words:
        return False
    if tag.startswith("NN"):
        return False
    return True


def predict(input_text, summary_list, pos_list, neg_list, logistic_reg_obj):
    features = feature_extractor(input_text, pos_list, neg_list)
    nb_probabilities = calculate(summary_list, [features['pos_words'], features['neg_words']])
    print(nb_probabilities)
    lg_prediction = logistic_reg_obj.hypothesis([features['pos_words'], features['neg_words']])
    print(lg_prediction)
    cnn_prediction = test_model(input_text)
    print(cnn_prediction)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nltk.download([
        "names",
        "stopwords",
        "state_union",
        "twitter_samples",
        "movie_reviews",
        "averaged_perceptron_tagger",
        "vader_lexicon",
        "punkt",
    ])

    pos_tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings(fileids=['positive_tweets.json'])]
    neg_tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings(fileids=['negative_tweets.json'])]
    file = open('Sentiment Analysis Dataset 100000.csv', 'r')
    file_reader = csv.reader(file)
    raw_data = []
    for row in file_reader:
        if row[1] == '1':
            pos_tweets.append(row[2].replace("://", "//"))
        else:
            neg_tweets.append(row[2].replace("://", "//"))

    unwanted_words = nltk.corpus.stopwords.words('english')
    unwanted_words.extend([w.lower() for w in nltk.corpus.names.words()])

    positive_tokenized = nltk.corpus.twitter_samples.tokenized(fileids=['positive_tweets.json'])
    negative_tokenized = nltk.corpus.twitter_samples.tokenized(fileids=['negative_tweets.json'])

    pos_words = list()
    neg_words = list()

    for token_list in positive_tokenized:
        word_list = [word for word, tag in filter(
            skip_unwanted,
            nltk.pos_tag(token_list)
        )]
        for word in word_list:
            pos_words.append(word.lower())

    for token_list in negative_tokenized:
        word_list = [word for word, tag in filter(
            skip_unwanted,
            nltk.pos_tag(token_list)
        )]
        for word in word_list:
            neg_words.append(word.lower())

    positive_fd = nltk.FreqDist(pos_words)
    negative_fd = nltk.FreqDist(neg_words)

    common_set = set(positive_fd).intersection(negative_fd)

    for word in common_set:
        del positive_fd[word]
        del negative_fd[word]

    common_pos = positive_fd.most_common(1500)
    common_neg = negative_fd.most_common(1500)

    file = open('dataset.csv', 'w')
    file_writer = csv.writer(file)
    for row in pos_tweets:
        feature = feature_extractor(row, common_pos, common_neg)
        feature['sentiment'] = 1
        file_writer.writerow([feature['pos_words'], feature['neg_words'], 1])
    for row in neg_tweets:
        feature = feature_extractor(row, common_pos, common_neg)
        feature['sentiment'] = 0
        file_writer.writerow([feature['pos_words'], feature['neg_words'], 0])
    file.close()

    file = open('dataset.csv', 'r')
    file_reader = csv.reader(file)
    dataset = []
    for row in file_reader:
        dataset.append([int(a) for a in row])

    summaries = summary(dataset[:1000])
    '''true_predicts = 0
    for data in dataset:
        probabilities = calculate(summaries, data)
        prediction = max(probabilities, key=probabilities.get)
        if prediction == data[2]:
            true_predicts += 1
    print(true_predicts)
    print(len(dataset))
    print(true_predicts/len(dataset))'''

    lg_true_predicts = 0
    lg = LogisticRegression(1, 0.3, dataset)
    lg.train()
    '''for data in dataset:
        prediction = round(lg.hypothesis(data))
        if prediction == data[2]:
            lg_true_predicts += 1
    print('result')
    print(lg_true_predicts/len(dataset))'''

    cnn = SimpleCNN(pos_tweets, neg_tweets, limit=2000)
    #cnn.train_model()
    '''length = len(pos_tweets) + len(neg_tweets)
    true = 0
    for data in pos_tweets[:100]:
        print('POSITIVE')
        prediction = test_model(data)
        if prediction == 'Positive':
            true += 1
    for data in neg_tweets[:100]:
        print('NEGATIVE')
        prediction = test_model(data)
        if prediction == 'Negative':
            true += 1
    print(true)
    print(length)
    print(true/200)'''
    schedule.every(10).seconds.do(predict, input_text='What a wonderful day!', summary_list=summaries,
                                  pos_list=common_pos, neg_list=common_neg, logistic_reg_obj=lg)

    while True:
        schedule.run_pending()
        time.sleep(1)
