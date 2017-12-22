#-*- coding=utf-8 -*-
import os
import collections
import numpy as np
from gensim.models import Word2Vec
import logging
import pickle

INPUT_PATH = '/home/zju/slx/binarylearning/bisheData/assemble_code/'

def build_dataset(dir_path,corp,labels):
    dir_list = os.listdir(dir_path)
    for dir in dir_list:
        file_list = os.listdir(INPUT_PATH+dir)
        for file_name in file_list:
            file_path = INPUT_PATH+dir+'/'+file_name
            f = open(file_path,mode='r')
            content = f.read().split('!')
            content = [line.split('$') for line in content]
            corp.append(content)
            labels.append(dir)
    return corp,labels

#将文档里的词用训练好的词向量代替，构成训练数据
def convert_document(documents,word_to_vec):
    doc_lenth=[]
    documents_flatten =[]
    documents_vector =[]
    max_lenth =0
    for doc in documents:
        doc_flatten = [word for setence in doc for word in setence]
        max_lenth = max(max_lenth,len(doc_flatten))
        documents_flatten.append(doc_flatten)

    # #RNN需要每个输入的序列长度一样，也就是每个文档包含同样多的词，如果不够则用0padding
    for doc in documents_flatten:
        doc_vector = np.zeros(shape=[max_lenth, word_to_vec[documents_flatten[0][0]].shape[-1]], dtype=np.float32)
        lenth = len(doc)
        doc_lenth.append(lenth)
        for i in range(lenth):
            doc_vector[i] = (word_to_vec[doc[i]])
        documents_vector.append(doc_vector)
    return documents_vector,documents_flatten,doc_lenth

def main():
    corpus = []
    labels = []
    print 'reading data...'
    corpus,labels = build_dataset(INPUT_PATH,corpus,labels)
    all_setences = [sentence for doc in corpus for sentence in doc]
    #flatten_corp = [word for line in all_setences for word in line]
    # count = collections.Counter(flatten_corp)
    # print 'vocabulary lenth: ',len(count)
    print 'start fitting word2vector '
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if os.path.exists('./ass2vec.model'):
        print 'model exists,loding...'
        model = Word2Vec.load('./ass2vec.model')
    else:
        print 'training model ...'
        model = Word2Vec(sentences=all_setences,workers=8,size=8,min_count=1)
        model.save('./ass2vec.model')                         #save model
    vocabulary_size = len(model.wv.vocab)
    print model.wv.most_similar('push')
    string_to_vector = {}
    i = 0
    for key in model.wv.vocab:
        string_to_vector[key] =model.wv.word_vec(key)
    vec_doc,flat_doc,doc_lenth = convert_document(corpus,string_to_vector)
    print len(corpus)
    print len(labels)
    print len(vec_doc),len(flat_doc),len(doc_lenth)
    print len(vec_doc[0])
    print len(flat_doc[0])
    print doc_lenth[0]
    #
    # #保存处理好的数据
    print 'save data_set...'
    data_file = open('./data/train_data.pkl',mode='wb')
    # label_file = open('./data/train_label.pkl',mode='wb')
    # lenth_file = open('./data/data_lenth.pkl',mode='wb')
    pickle.dump(vec_doc,data_file,protocol=1)
    # pickle.dump(labels,label_file,protocol=1)
    # pickle.dump(doc_lenth,lenth_file,protocol=1)
    print 'complete saved ! '
    data_file.close()
        # ,label_file.close(),lenth_file.close()

if __name__ == '__main__':
    main()
    # print 'check data ...'
    # data_file = open('./train_data.pkl',mode='r')
    # label_file = open('./train_label.pkl',mode='r')
    # lenth_file = open('./data_lenth.pkl',mode='r')
    # train_data = pickle.load(data_file)
    # train_label = pickle.load(label_file)
    # data_lenth = pickle.load(lenth_file)
    # print len(train_data),len(train_label),len(data_lenth)
    # print train_data[0].shape
    # print train_label[0]
    # print data_lenth[0]

