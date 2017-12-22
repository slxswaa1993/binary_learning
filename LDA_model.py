# -*- coding=utf-8 -*-
import os
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
import numpy as np
import random
import data_helper
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import recall_score,precision_score,accuracy_score
from sklearn import tree
import pandas as pd
import graphviz

INPUT_PATH='/home/zju/slx/binarylearning/bisheData/newData/'



def tf_idf_model(train_data,test_data):
    tfidf_vector = TfidfVectorizer()
    tfidf_train = tfidf_vector.fit_transform(train_data)
    tfidf_test = tfidf_vector.transform(test_data)
    print tfidf_train.shape,tfidf_test.shape
    return tfidf_train,tfidf_test

def lda_model(train_data,test_data):
    tf_vectorizer = CountVectorizer()
    tf_train_vector = tf_vectorizer.fit_transform(train_data)
    tf_test_vector = tf_vectorizer.transform(test_data)
    lda = LatentDirichletAllocation(n_components=35,learning_method='batch')
    doc_topic_distribution = lda.fit_transform(tf_train_vector)
    test_doc_topic_distribution = lda.transform(tf_test_vector)
    return lda,doc_topic_distribution,test_doc_topic_distribution

def lda_model_for_asseble(corpus):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word')
    tfidf_vector = tfidf_vectorizer.fit_transform(corpus)

    tf_vectorizer = CountVectorizer()
    tf_vector = tf_vectorizer.fit_transform(corpus)

    lda = LatentDirichletAllocation(n_components=35, learning_method='batch')
    doc_topic_distribution = lda.fit_transform(tf_vector)
    return lda,doc_topic_distribution

def evaluate(estimator,test_x,test_y):
    pred_y = estimator.predict(test_x)
    recall = recall_score(y_true=test_y,y_pred=pred_y,average='macro')
    precision = precision_score(y_true=test_y,y_pred=pred_y,average='macro')
    acc = accuracy_score(y_true=test_y,y_pred=pred_y)
    print 'accuracy recall precision is {0} {1} {2}'.format(acc,recall,precision)

def train(train_path,test_path,is_bytecode):
    print 'loading data'
    train_x ,train_y = data_helper.prepare_classification_data(train_path,is_bytecode)
    test_x,test_y = data_helper.prepare_classification_data(test_path,is_bytecode)
    print 'start fit LDA model'
    lda_train,train_doc_topic,test_doc_topic = lda_model(train_x,test_x)
    print 'complete fitting LDA model,using DOC-TOPIC feature to decision tree '
    DT = DecisionTreeClassifier()
    DT.fit(train_doc_topic,train_y)
    print 'COMPLETE FITTING DECISION TREE '
    #dot = export_graphviz(DT,out_file='./tree.dot')
    evaluate(DT,test_doc_topic,test_y)


def eval_model(real, predict, name):
    assert len(predict) == len(real)
    print name
    print 'crosstab:{0}'.format(pd.crosstab(np.asarray(real), np.asarray(predict), margins=True))
    print 'precision:{0}'.format(precision_score(real, predict, average='macro'))
    print 'recall:{0}'.format(recall_score(real, predict, average='macro'))
    print 'accuracy:{0}'.format(accuracy_score(real, predict))

def train_tfidf(train_path, test_path, is_bytecode):
    train_x, train_y = data_helper.prepare_classification_data(train_path, is_bytecode)
    test_x, test_y = data_helper.prepare_classification_data(test_path, is_bytecode)

    print len(train_y)
    print len(train_x)
    tfidf_train,tfidf_test = tf_idf_model(train_x,test_x)

    # clf = MultinomialNB().fit(x_train, y_train)
    clf = tree.DecisionTreeClassifier().fit(tfidf_train, train_y)

    evaluate(estimator=clf,test_x=tfidf_test,test_y=test_y)
    return clf



if __name__ == '__main__':
    TRAIN_INPUT = INPUT_PATH+'train.txt'
    TEST_INPUT = INPUT_PATH+'test.txt'
    DEV_INPUT = INPUT_PATH+'dev.txt'
    train_tfidf(TRAIN_INPUT,TEST_INPUT,True)
