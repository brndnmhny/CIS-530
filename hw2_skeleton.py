training = "C:/Users/brend/Downloads/data/complex_words_training.txt"
development = "C:/Users/brend/Downloads/data/complex_words_development.txt"
ngram = "C:/Users/brend/Downloads/ngram_counts.txt.gz"
testing = "C:/Users/brend/Downloads/data/complex_words_test_unlabeled.txt"
import matplotlib.pyplot as mat
import re
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import gzip
import collections
#from keras.models import Sequential
#from keras.layers import Dense
import nltk
nltk.download('averaged_perceptron_tagger', 'wordnet')
from nltk.corpus import wordnet as wn


# Answer 1

def get_precision(y_pred, y_true):
    tp = fp = 0
    for i, j in zip(y_pred, y_true):
        if i == 1 and j == 1:
            tp += 1
        elif i == 1 and j == 0:
            fp += 1
    if tp == 0 and fp == 0:
        precision = "NA"
    else: precision = tp / (tp + fp)
    return precision

def get_recall(y_pred, y_true):
    tp = fn = 0
    for i, j in zip(y_pred, y_true):
        if i == 1 and j == 1:
            tp += 1
        elif i == 0 and j == 1:
            fn += 1
    if tp == 0 and fn == 0:
        recall = "NA"
    else: recall = tp / (tp + fn)
    return recall

def get_fscore(y_pred, y_true):
    beta = 1
    tp = fp = fn = 0
    for i, j in zip(y_pred, y_true):
        if i == 1 and j == 1:
            tp += 1
        elif i == 1 and j == 0:
            fp += 1
        elif i == 0 and j == 1:
            fn += 1
    if tp == 0 and fp == 0:
        fscore = "NA"
    elif tp == 0 and fn == 0:
        fscore = "NA"
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = (((beta ^ 2) + 1) * precision * recall) / (((beta ^ 2) *precision) + recall)
    return fscore

def get_predictions(y_pred, y_true):
    precision = print(get_precision(y_pred, y_true))
    recall = print(get_recall(y_pred, y_true))
    fscore = print(get_fscore(y_pred, y_true))
    return precision, recall, fscore

# Answer 2

def load_file(data_file):
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

print("Answer 2.1")

def all_complex(data_file):
    words, labels = load_file(data_file)
    y_pred = [1] * len(labels)
    get_predictions(y_pred, labels)

print("Training Set Performance:")
all_complex(training)
print("Development Set Performance:")
all_complex(development)

print("\nAnswer 2.2")

def word_length_threshold(training_file, development_file):
    words_train, labels_train = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    fscores = []
    precision_len = []
    recall_len = []
    for t in range(1,20):
        y_pred_train = []
        for w in words_train:
            if len(w) >= t:
                y_pred_train.append(1)
            else:
                y_pred_train.append(0)
        p = get_precision(y_pred_train, labels_train)
        r = get_recall(y_pred_train, labels_train)
        f = get_fscore(y_pred_train, labels_train)
        fscores.append((f, t))
        precision_len.append(p)
        recall_len.append(r)
    best = max(fscores)
    print("In thresholds 1 through 20, the best threshold is", best[1], " which produces an f-score of", best[0])

    mat.plot(recall_len, precision_len, color = 'blue')
    mat.title('Word Length')
    mat.xlabel('Recall')
    mat.ylabel('Precision')
    mat.show()

    y_pred_train = []
    y_pred_dev = []
    for w in words_train:
        if len(w) >= 6:
            y_pred_train.append(1)
        else:
            y_pred_train.append(0)
    for w in words_dev:
        if len(w) >= 6:
            y_pred_dev.append(1)
        else:
            y_pred_dev.append(0)
    training_performance = get_predictions(y_pred_train, labels_train)
    development_performance = get_predictions(y_pred_dev, labels_dev)
    return training_performance, development_performance, recall_len, precision_len

word_length_threshold(training, development)

print("\nAnswer 2.3")

def load_ngram_counts(ngram_counts_file):
   counts = collections.defaultdict(int)
   with gzip.open(ngram_counts_file, 'rt', encoding= 'utf-8', errors='ignore') as f:
       for line in f:
           token, count = line.strip().split('\t')
           if token[0].islower():
               counts[token] = int(count)
   return counts

def word_frequency_threshold(training_file, development_file, counts):
    words_train, labels_train = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    counts = load_ngram_counts(counts)

    thresholds = range(1000000, 100000000, 100000)
    precision_freq, recall_freq, fscores = [], [], []
    for threshold in thresholds:
        y_pred_train = []
        for word_train in words_train:
            if counts[word_train] < threshold:
                y_pred_train.append(1)
            else:
                y_pred_train.append(0)
        precision_freq.append(get_precision(y_pred_train, labels_train))
        recall_freq.append(get_recall(y_pred_train, labels_train))
        fscores.append((get_fscore(y_pred_train, labels_train), threshold))

    mat.plot(recall_freq, precision_freq, color = 'red')
    mat.title('Word Frequency')
    mat.xlabel('Recall')
    mat.ylabel('Precision')
    mat.show()

    best = max(fscores)
    print("In thresholds 1 through 100,000,000, the best threshold is", best[1], " which produces an f-score of",
          best[0])

    y_pred_train = []
    y_pred_dev = []
    for w in words_train:
        if counts[w] < best[1]:
            y_pred_train.append(1)
        else:
            y_pred_train.append(0)
    for w in words_dev:
        if counts[w] < best[1]:
            y_pred_dev.append(1)
        else:
            y_pred_dev.append(0)

    training_performance = get_predictions(y_pred_train, labels_train)
    development_performance = get_predictions(y_pred_dev, labels_dev)
    return training_performance, development_performance, recall_freq, precision_freq


word_frequency_threshold(training, development, ngram)

l, m, recall_len, precision_len = word_length_threshold(training, development)
l, m, recall_freq, precision_freq = word_frequency_threshold(training, development, ngram)

mat.plot(recall_len, precision_len, "r", recall_freq, precision_freq, "b")
mat.title('Comparing Two Classifiers')
mat.xlabel('Recall')
mat.ylabel('Precision')
mat.show()

print("\nAnswer 3.1")

def naive_bayes(training_file, development_file, counts):
    counts = load_ngram_counts(counts)
    words_train, labels_train = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    X_train = []
    X_dev = []
    Z_train = []
    Z_dev = []


    for w in words_train:
        if len(w) >= 6:
            X_train.append(1)
        else:
            X_train.append(0)
    for w in words_dev:
        if len(w) >= 6:
            X_dev.append(1)
        else:
            X_dev.append(0)

    for w in words_train:
        if counts[w] < 69099000:
            Z_train.append(1)
        else:
            Z_train.append(0)
    for w in words_dev:
        if counts[w] < 69099000:
            Z_dev.append(1)
        else:
            Z_dev.append(0)

    array_train = np.array([X_train, Z_train]).T
    array_dev = np.array([X_dev, Z_dev]).T
    labels_train = np.array(labels_train)
    labels_dev = np.array(labels_dev)

    gnb = GaussianNB()
    gnb.fit(array_train, labels_train)
    y_pred_train = gnb.predict(array_train)
    y_pred_dev = gnb.predict(array_dev)
    training_performance = get_predictions(y_pred_train, labels_train)
    development_performance = get_predictions(y_pred_dev, labels_dev)
    return development_performance

naive_bayes(training, development, ngram)

print("\nAnswer 3.2")

def logistic_regression(training_file, development_file, counts):
    counts = load_ngram_counts(counts)
    words_train, labels_train = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    X_train = []
    X_dev = []
    Z_train = []
    Z_dev = []


    for w in words_train:
        if len(w) >= 6:
            X_train.append(1)
        else:
            X_train.append(0)
    for w in words_dev:
        if len(w) >= 6:
            X_dev.append(1)
        else:
            X_dev.append(0)

    for w in words_train:
        if counts[w] < 69099000:
            Z_train.append(1)
        else:
            Z_train.append(0)
    for w in words_dev:
        if counts[w] < 69099000:
            Z_dev.append(1)
        else:
            Z_dev.append(0)

    array_train = np.array([X_train, Z_train]).T
    array_dev = np.array([X_dev, Z_dev]).T
    labels_train = np.array(labels_train)
    labels_dev = np.array(labels_dev)

    lr = LogisticRegression()
    lr.fit(array_train, labels_train)
    y_pred_train = lr.predict(array_train)
    y_pred_dev = lr.predict(array_dev)
    training_performance = get_predictions(y_pred_train, labels_train)
    development_performance = get_predictions(y_pred_dev, labels_dev)
    return development_performance

logistic_regression(training, development, ngram)

print("\nAnswer 3.3:\nWell my results for these tests are exactly the same so I assume I've done something wrong here, but, as per\n"
      "the assignment, if they WERE different I would probably attempt to explain it by talking about how these two\n"
      "classifiers are looking at different things. The Naive Bayes is a generative classifier whereas the \n"
      "logistic regression is a discriminative classifier, which I would think probably makes the LR better at\n"
      "a task like this.\n\nAnswer 4:")


def lemma(word_pos):
    lemma, pos = word_pos
    lemmatizer = nltk.stem.WordNetLemmatizer()
    if pos[:2] == 'NN':
        return lemmatizer.lemmatize(lemma, pos='n')
    elif pos[:2] == 'JJ':
        return lemmatizer.lemmatize(lemma, pos='a')
    elif pos[0] == 'V':
        return lemmatizer.lemmatize(lemma, pos='v')
    elif pos[:2] == 'RB':
        try:
            return nltk.corpus.wordnet.synset(lemma +'.r.1').lemmas()[0].pertainyms()[0].name()
        except BaseException:
            return lemmatizer.lemmatize(lemma, pos='r')
    else:
        return lemmatizer.lemmatize(lemma)

def count_syllables(word):
    word = word.lower()
    # exception_add are words that need extra syllables
    # exception_del are words that need less syllables
    exception_add = ['serious', 'crucial']
    exception_del = ['fortunately', 'unfortunately']
    co_one = ['cool', 'coach', 'coat', 'coal', 'count', 'coin', 'coarse', 'coup', 'coif', 'cook', 'coign', 'coiffe',
              'coof', 'court']
    co_two = ['coapt', 'coed', 'coinci']
    pre_one = ['preach']

    syls = 0  # added syllable number
    disc = 0  # discarded syllable number

    # 1) if letters < 3 : return 1
    if len(word) <= 3:
        syls = 1
        return syls

    # 2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
    # if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.)
    if word[-2:] == "es" or word[-2:] == "ed":
        doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]', word))
        if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]', word)) > 1:
            if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[
                                                                                                       -3:] == "ies":
                pass
            else:
                disc += 1

    # 3) discard trailing "e", except where ending is "le"
    le_except = ['whole', 'mobile', 'pole', 'male', 'female', 'hale', 'pale', 'tale', 'sale', 'aisle', 'whale', 'while']
    if word[-1:] == "e":
        if word[-2:] == "le" and word not in le_except:
            pass
        else:
            disc += 1

    # 4) check if consecutive vowels exists, triplets or pairs, count them as one.
    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]', word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]', word))
    disc += doubleAndtripple + tripple

    # 5) count remaining vowels in word.
    numVowels = len(re.findall(r'[eaoui]', word))

    # 6) add one if starts with "mc"
    if word[:2] == "mc":
        syls += 1

    # 7) add one if ends with "y" but is not surrouned by vowel
    if word[-1:] == "y" and word[-2] not in "aeoui":
        syls += 1

    # 8) add one if "y" is surrounded by non-vowels and is not in the last word.
    for i, j in enumerate(word):
        if j == "y":
            if (i != 0) and (i != len(word) - 1):
                if word[i - 1] not in "aeoui" and word[i + 1] not in "aeoui":
                    syls += 1

    # 9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.
    if word[:3] == "tri" and word[3] in "aeoui":
        syls += 1
    if word[:2] == "bi" and word[2] in "aeoui":
        syls += 1

    # 10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"
    if word[-3:] == "ian":
        # and (word[-4:] != "cian" or word[-4:] != "tian") :
        if word[-4:] == "cian" or word[-4:] == "tian":
            pass
        else:
            syls += 1

    # 11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.
    if word[:2] == "co" and word[2] in 'eaoui':
        if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two:
            syls += 1
        elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one:
            pass
        else:
            syls += 1

    # 12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.
    if word[:3] == "pre" and word[3] in 'eaoui':
        if word[:6] in pre_one:
            pass
        else:
            syls += 1

    # 13) check for "-n't" and cross match with dictionary to add syllable.
    negative = ["doesn't", "isn't", "shouldn't", "couldn't", "wouldn't"]
    if word[-3:] == "n't":
        if word in negative:
            syls += 1
        else:
            pass

            # 14) Handling the exceptional words.
    if word in exception_del:
        disc += 1
    if word in exception_add:
        syls += 1

        # calculate the output
    return numVowels - disc + syls

#Gonna try to build a neural net ¯\_(ツ)_/¯

#def neuralnet(training_file, development_file, counts):
#    counts = load_ngram_counts(counts)
#    words_train, labels_train = load_file(training_file)
#    words_dev, labels_dev = load_file(development_file)
#    pos_train = nltk.pos_tag(words_train)
#    pos_dev = nltk.pos_tag(words_dev)
#
#    f1_train = np.array([len(word) for word in words_train])
#    f2_train = np.array([counts[word] for word in words_train])
#    f3_train = np.array([len(wn.synsets(word)) for word in words_train])
#    f4_train = np.array([count_syllables(word) for word in words_train])
#    f5_train = np.array([len(lemma(word_pos)) for word_pos in pos_train])
#    f6_train = np.array([counts[lemma(word_pos)] for word_pos in pos_train])
#    f7_train = np.array([len(wn.synsets(lemma(word_pos))) for word_pos in pos_train])
#    f8_train = np.array([count_syllables(lemma(word_pos)) for word_pos in pos_train])
#    array_train = np.column_stack( (f1_train, f2_train, f3_train, f4_train, f5_train, f6_train, f7_train, f8_train))
#
#
#    f1_dev = np.array([len(word) for word in words_dev])
#    f2_dev = np.array([counts[word] for word in words_dev])
#    f3_dev = np.array([len(wn.synsets(word)) for word in words_dev])
#    f4_dev = np.array([count_syllables(word) for word in words_dev])
#    f5_dev = np.array([len(lemma(word_pos)) for word_pos in pos_dev])
#    f6_dev = np.array([counts[lemma(word_pos)] for word_pos in pos_dev])
#    f7_dev = np.array([len(wn.synsets(lemma(word_pos))) for word_pos in pos_dev])
#    f8_dev = np.array([count_syllables(lemma(word_pos)) for word_pos in pos_dev])
#    array_dev = np.column_stack((f1_dev, f2_dev, f3_dev, f4_dev, f5_dev, f6_dev, f7_dev, f8_dev))
#
#
#    labels_train = np.array(labels_train)
#    labels_dev = np.array(labels_dev)
#
#    fscores = []
#    i = 0
#    while i <= 20:
#        model = Sequential()
#        model.add(Dense(units = 64, activation = 'relu', input_dim = 8))
#        model.add(Dense(units = 64, activation = 'relu'))
#        model.add(Dense(units = 64, activation = 'relu'))
#        model.add(Dense(units = 1, activation = 'sigmoid'))
#        model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
#        model.fit(array_train, labels_train, epochs = 20, batch_size = 512)
#        y_pred = model.predict(array_dev)
#        f = get_fscore(y_pred, labels_dev)
#        if f != "NA":
#            fscores.append((f, model))
#        i += 1
#    best = max(fscores)
#    y_pred = best[1].predict(array_dev)
#    np.savetxt("C:/Users/brend/Documents/CIS 530/HW 2/test_labels.txt", y_pred)
#    development_performance = get_predictions(y_pred, labels_dev)
#    return development_performance


#print("Neural Net Performance:")
#neuralnet(training, development, ngram)

def randomforest(training_file, development_file, counts):
    counts = load_ngram_counts(counts)
    words_train, labels_train = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    pos_train = nltk.pos_tag(words_train)
    pos_dev = nltk.pos_tag(words_dev)

    f1_train = np.array([len(word) for word in words_train])
    f2_train = np.array([counts[word] for word in words_train])
    f3_train = np.array([len(wn.synsets(word)) for word in words_train])
    f4_train = np.array([count_syllables(word) for word in words_train])
    f5_train = np.array([len(lemma(word_pos)) for word_pos in pos_train])
    f6_train = np.array([counts[lemma(word_pos)] for word_pos in pos_train])
    f7_train = np.array([len(wn.synsets(lemma(word_pos))) for word_pos in pos_train])
    f8_train = np.array([count_syllables(lemma(word_pos)) for word_pos in pos_train])
    X_train = np.column_stack((f1_train, f2_train, f3_train, f4_train, f5_train, f6_train, f7_train, f8_train))
    scaler_train = preprocessing.StandardScaler().fit(X_train)
    array_train = scaler_train.transform(X_train)

    f1_dev = np.array([len(word) for word in words_dev])
    f2_dev = np.array([counts[word] for word in words_dev])
    f3_dev = np.array([len(wn.synsets(word)) for word in words_dev])
    f4_dev = np.array([count_syllables(word) for word in words_dev])
    f5_dev = np.array([len(lemma(word_pos)) for word_pos in pos_dev])
    f6_dev = np.array([counts[lemma(word_pos)] for word_pos in pos_dev])
    f7_dev = np.array([len(wn.synsets(lemma(word_pos))) for word_pos in pos_dev])
    f8_dev = np.array([count_syllables(lemma(word_pos)) for word_pos in pos_dev])
    X_dev = np.column_stack((f1_dev, f2_dev, f3_dev, f4_dev, f5_dev, f6_dev, f7_dev, f8_dev))
    scaler_dev = preprocessing.StandardScaler().fit(X_dev)
    array_dev = scaler_dev.transform(X_dev)

    labels_train = np.array(labels_train)
    labels_dev = np.array(labels_dev)
    words_dev = np.array(words_dev)

    clf = RandomForestClassifier()
    clf.fit(array_train, labels_train)

    y_pred = clf.predict(array_dev)

    development_performance = get_predictions(y_pred, labels_dev)
    return development_performance

print("Random Forest Performance:")
randomforest(training, development, ngram)

def randomforestfinal(training_file, development_file, testing_file, counts):
    counts = load_ngram_counts(counts)
    words_train, labels_train = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    words_test = [line.split()[0] for line in open(testing_file)]
    pos_test = nltk.pos_tag(words_test)
    words = words_train + words_dev
    labels = labels_train + labels_dev
    pos = nltk.pos_tag(words)

    f1_train = np.array([len(word) for word in words])
    f2_train = np.array([counts[word] for word in words])
    f3_train = np.array([len(wn.synsets(word)) for word in words])
    f4_train = np.array([count_syllables(word) for word in words])
    f5_train = np.array([len(lemma(word_pos)) for word_pos in pos])
    f6_train = np.array([counts[lemma(word_pos)] for word_pos in pos])
    f7_train = np.array([len(wn.synsets(lemma(word_pos))) for word_pos in pos])
    f8_train = np.array([count_syllables(lemma(word_pos)) for word_pos in pos])
    X_train = np.column_stack((f1_train, f2_train, f3_train, f4_train, f5_train, f6_train, f7_train, f8_train))
    scaler_train = preprocessing.StandardScaler().fit(X_train)
    array_train = scaler_train.transform(X_train)

    f1_test = np.array([len(word) for word in words_test])
    f2_test = np.array([counts[word] for word in words_test])
    f3_test = np.array([len(wn.synsets(word)) for word in words_test])
    f4_test = np.array([count_syllables(word) for word in words_test])
    f5_test = np.array([len(lemma(word_pos)) for word_pos in pos_test])
    f6_test = np.array([counts[lemma(word_pos)] for word_pos in pos_test])
    f7_test = np.array([len(wn.synsets(lemma(word_pos))) for word_pos in pos_test])
    f8_test = np.array([count_syllables(lemma(word_pos)) for word_pos in pos_test])
    X_test = np.column_stack((f1_test, f2_test, f3_test, f4_test, f5_test, f6_test, f7_test, f8_test))
    scaler_test = preprocessing.StandardScaler().fit(X_test)
    array_test = scaler_test.transform(X_test)

    labels = np.array(labels)

    clf = RandomForestClassifier()
    clf.fit(array_train, labels)

    y_pred = clf.predict(array_test)

    with open('C:/Users/brend/Documents/CIS 530/HW 2/test_labels.txt', 'w') as f:
        for label in y_pred[1:]:
            f.write(str(label))
            f.write('\n')
        f.close()

randomforestfinal(training, development, testing, ngram)

#def neuralnetfinal(training_file, development_file, testing_file, counts):
#    counts = load_ngram_counts(counts)
#    words_train, labels_train = load_file(training_file)
#    words_dev, labels_dev = load_file(development_file)
#    words_test = [line.split()[0] for line in open(testing_file)]
#    pos_test = nltk.pos_tag(words_test)
#    words = words_train + words_dev
#    labels = labels_train + labels_dev
#    pos = nltk.pos_tag(words)
#
#    f1_train = np.array([len(word) for word in words])
#    f2_train = np.array([counts[word] for word in words])
#    f3_train = np.array([len(wn.synsets(word)) for word in words])
#    f4_train = np.array([count_syllables(word) for word in words])
#    f5_train = np.array([len(lemma(word_pos)) for word_pos in pos])
#    f6_train = np.array([counts[lemma(word_pos)] for word_pos in pos])
#    f7_train = np.array([len(wn.synsets(lemma(word_pos))) for word_pos in pos])
#    f8_train = np.array([count_syllables(lemma(word_pos)) for word_pos in pos])
#    array_train = np.column_stack((f1_train, f2_train, f3_train, f4_train, f5_train, f6_train, f7_train, f8_train))
#
#    f1_test = np.array([len(word) for word in words_test])
#    f2_test = np.array([counts[word] for word in words_test])
#    f3_test = np.array([len(wn.synsets(word)) for word in words_test])
#    f4_test = np.array([count_syllables(word) for word in words_test])
#    f5_test = np.array([len(lemma(word_pos)) for word_pos in pos_test])
#    f6_test = np.array([counts[lemma(word_pos)] for word_pos in pos_test])
#    f7_test = np.array([len(wn.synsets(lemma(word_pos))) for word_pos in pos_test])
#    f8_test = np.array([count_syllables(lemma(word_pos)) for word_pos in pos_test])
#    array_test = np.column_stack((f1_test, f2_test, f3_test, f4_test, f5_test, f6_test, f7_test, f8_test))
#
#    labels = np.array(labels)
#
#    model = Sequential()
#    model.add(Dense(units = 64, activation = 'relu', input_dim = 8))
#    model.add(Dense(units = 64, activation = 'relu'))
#    model.add(Dense(units = 64, activation = 'relu'))
#    model.add(Dense(units = 1, activation = 'sigmoid'))
#    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
#    model.fit(array_train, labels, epochs = 60, batch_size = 512)
#    y_pred = model.predict(array_test)
#    y_pred = np_where(y_pred > .2, 1, 0)
#
#    with open('C:/Users/brend/Documents/CIS 530/HW 2/test_labels.txt', 'w') as f:
#        for label in y_pred[1:]:
#            f.write(str(label))
#            f.write('\n')
#        f.close()
#
#neuralnetfinal(training, development, testing, ngram)