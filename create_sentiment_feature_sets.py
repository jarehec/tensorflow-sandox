import nltk
import numpy as np
import random
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
n_lines = 10000000

def create_lexicon(pos, neg):
    lexicon = []
    for f in [pos, neg]:
        with open(f, 'r') as data:
            contents = data.readlines()
            for l in contents[:n_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    l2 = []
    w_counts = Counter(lexicon)
    for w in w_counts:
        # should be a percentage, not hardcoded
        if 50 < w_counts[w] < 1000:
            l2.append(w)
    print(len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    feature_set = []

    with open(sample, 'r') as data:
        contents = data.readlines()
        for l in contents[:n_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))

            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            feature_set.append([features, classification])
    return feature_set

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0,1])
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_size * len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)