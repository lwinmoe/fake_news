#!/usr/bin/python3

import string
import itertools

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk import tokenize

from wordcloud import WordCloud

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn. feature_extraction. text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score

import numpy as np


token_space = tokenize.WhitespaceTokenizer()

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()


def explore_data(data):

    # group by
    group_data(data)

    #word cloud
    word_cloud(data)

    # frequent words
    counter(data[data['target'] == 'fake'], "text", 20)


def group_data(data):
    print(data.groupby(['subject'])['text'].count())
    data.groupby(['subject'])['text'].count().plot(kind="bar")
    plt.show()

    print(data.groupby(['target'])['text'].count())
    data.groupby(['target'])['text'].count().plot(kind="bar")
    plt.show()


def word_cloud(data):
    fake_data = data[data["target"] == "fake"]
    all_words = ' '.join([text for text in fake_data.text])
    wordcloud = WordCloud(width= 800, height= 500,
                            max_font_size = 110,
                            collocations = False).generate(all_words)
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    real_data = data[data["target"] == "true"]
    all_words = ' '.join([text for text in real_data.text])
    wordcloud = WordCloud(width= 800, height= 500,
                            max_font_size = 110,
                            collocations = False).generate(all_words)
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def main():
    fake = pd.read_csv("data/Fake.csv")
    #print(fake)
    #print("========================")

    true = pd.read_csv("data/True.csv")
    #print(true)
    #print("========================")

    fake['target'] = 'fake'
    true['target'] = 'true'

    # merge
    data = pd.concat([fake, true]).reset_index(drop = True)


    # shuffle data
    data = shuffle(data)
    data = data.reset_index(drop=True)

    # drop the date column and title column
    data.drop(["date"],axis=1,inplace=True)
    data.drop(["title"],axis=1,inplace=True)

    # to lower case
    data['text'] = data['text'].apply(lambda x: x.lower())

    # remove punctuation
    data['text'] = data['text'].apply(punctuation_removal)

    # remove stopwords
    stop = stopwords.words('english')
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    # data exploration
    #explore_data(data)

    #split the data
    X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)



    ############## logistics regression
    pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])
    # Fitting the model
    model = pipe.fit(X_train, y_train)
    # Accuracy
    prediction = model.predict(X_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

    #confusion matrix
    cm = metrics.confusion_matrix(y_test, prediction)
    plot_confusion_matrix(cm, classes=['Fake', 'Real'])



    ################# decision tree classifier
    # Vectorizing and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('model', DecisionTreeClassifier(criterion= 'entropy',
                                            max_depth = 20,
                                            splitter='best',
                                            random_state=42))])
    # Fitting the model
    model = pipe.fit(X_train, y_train)
    # Accuracy
    prediction = model.predict(X_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

    cm = metrics.confusion_matrix(y_test, prediction)
    plot_confusion_matrix(cm, classes=['Fake', 'Real'])



    ################# Random Forest Classifier
    pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
    model = pipe.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

    cm = metrics.confusion_matrix(y_test, prediction)
    plot_confusion_matrix(cm, classes=['Fake', 'Real'])



if __name__ == "__main__":
    main()
