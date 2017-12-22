import pandas as pd
import numpy as np
import os

INPUT_PATH='/home/zju/slx/binarylearning/bisheData/newData/'

def prepare_classification_data(data_path,is_bytecode):

    df = pd.read_csv(data_path, sep='@', header=None, encoding='utf8', engine='python')
    selected = ['tag', 'assemble','byte']
    df.columns = selected
    if(is_bytecode):
        texts = df[selected[2]].values.astype('U')
    else:
        texts = df[selected[1]]
        #texts = [s.encode('utf-8') for s in texts]
    labels = df[selected[0]].tolist()
    return texts,labels

def build_assemble_data(data_path):
    corpus = []
    labels = []
    dir_list = os.listdir(data_path)
    for dir in dir_list:
        dir_path = os.path.join(data_path,dir)
        current_label = dir
        file_list = os.listdir(dir_path)
        for file_name in file_list:
            file_path = os.path.join(dir_path,file_name)
            f = open(file_path,mode='r')
            content = ' '.join(f.read().split('!'))
            corpus.append(content)
            labels.append(current_label)
    return corpus,labels


