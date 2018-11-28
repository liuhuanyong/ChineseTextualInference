#!/usr/bin/env python3
# coding: utf-8
# File: siamese_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam,SGD
from keras.utils import to_categorical, plot_model
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Dense, Input, Dropout, Reshape, BatchNormalization, TimeDistributed, Lambda, Layer, LSTM, Bidirectional, Average, concatenate
import matplotlib.pyplot as plt
import os
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SiameseNetwork:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.class_dict ={
                         'neutral':0,
                         'entailment': 1,
                         'contradiction': 2,
                         }
        self.train_path = os.path.join(cur, 'data/train.txt')
        self.test_path = os.path.join(cur, 'data/test.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'tokenvec_bilstm2_model.h5')
        self.datas, self.word_dict = self.build_data()
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 20
        self.BATCH_SIZE = 512
        self.LIMIT_RATE = 0.95
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = self.select_best_length()
        self.embedding_matrix = self.build_embedding_matrix()

    '''根据样本长度,选择最佳的样本max-length'''
    def select_best_length(self):
        len_list = []
        max_length = 0
        cover_rate = 0.0
        sent_list = set()
        for line in open(self.train_path):
            line = line.strip().split('\t')
            if len(line) < 3:
                continue
            sent1 = line[0]
            sent2 = line[1]
            sent_list.add(sent1)
            sent_list.add(sent2)

        for sent in sent_list:
            sent_len = len(sent)
            len_list.append(sent_len)
        all_sent = len(len_list)
        sum_length = 0
        len_dict = Counter(len_list).most_common()
        for i in len_dict:
            sum_length += i[1] * i[0]
        average_length = sum_length / all_sent
        for i in len_dict:
            rate = i[1] / all_sent
            cover_rate += rate
            if cover_rate >= self.LIMIT_RATE:
                max_length = i[0]
                break
        print('average_length:', average_length)
        print('max_length:', max_length)
        return max_length

    '''构造数据集'''
    def build_data(self):
        sample_y = []
        sample_x_left = []
        sample_x_right = []
        vocabs = {'UNK'}
        count = 0
        for line in open(self.train_path):
            line = line.rstrip().split('\t')
            if not line or len(line)<3:
                continue
            sent_left = line[0]
            sent_right = line[1]
            label = line[2]
            if label not in self.class_dict:
                continue
            sample_x_left.append([char for char in sent_left if char])
            sample_x_right.append([char for char in sent_right if char])
            sample_y.append(label)
            for char in [char for char in sent_left + sent_right if char]:
                vocabs.add(char)
            count += 1
            if count%10000 == 0:
                print(count)
        print(len(sample_x_left), len(sample_x_right))
        sample_x = [sample_x_left, sample_x_right]
        datas = [sample_x, sample_y]
        word_dict = {wd:index for index, wd in enumerate(list(vocabs))}
        self.write_file(list(vocabs), self.vocab_path)
        return datas, word_dict

    '''将数据转换成keras所需的格式'''
    def modify_data(self):
        sample_x = self.datas[0]
        sample_y = self.datas[1]
        sample_x_left = sample_x[0]
        sample_x_right = sample_x[1]
        left_x_train = [[self.word_dict[char] for char in data] for data in sample_x_left]
        right_x_train = [[self.word_dict[char] for char in data] for data in sample_x_right]
        y_train = [self.class_dict.get(i) for i in sample_y]
        left_x_train = pad_sequences(left_x_train, self.TIME_STAMPS)
        right_x_train = pad_sequences(right_x_train, self.TIME_STAMPS)
        y_train = to_categorical(y_train, num_classes=3)
        return left_x_train, right_x_train, y_train

    '''保存字典文件'''
    def write_file(self, wordlist, filepath):
        with open(filepath, 'w+') as f:
            f.write('\n'.join(wordlist))

    '''加载预训练词向量'''
    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, 'r') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    '''加载词向量矩阵'''
    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    '''搭建编码层网络,用于权重共享'''
    def create_base_network(self, input_shape):
        input = Input(shape=input_shape)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
        lstm1 = Dropout(0.5)(lstm1)
        lstm2 = Bidirectional(LSTM(64))(lstm1)
        lstm2 = Dropout(0.5)(lstm2)
        return Model(input, lstm2)

    '''搭建网络'''
    def bilstm_siamese_model(self):
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)
        left_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        right_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)
        shared_lstm = self.create_base_network(input_shape=(self.TIME_STAMPS, self.EMBEDDING_DIM))
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)
        merged = concatenate([left_output, right_output], axis=-1)
        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)
        pred = Dense(self.NUM_CLASSES, activation='softmax', name='softmax_prediction')(merged)
        optimizer = SGD(lr=0.001, momentum=0.9)
        model = Model(inputs=[left_input, right_input], outputs=pred)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        return model

    '''训练模型'''
    def train_model(self):
        left_x_train, right_x_train, y_train = self.modify_data()
        model = self.bilstm_siamese_model()
        history = model.fit(
                              x=[left_x_train, right_x_train],
                              y=y_train,
                              validation_split=0.25,
                              batch_size=self.BATCH_SIZE,
                              epochs=self.EPOCHS,
                            )
        self.draw_train(history)
        model.save(self.model_path)
        return model

    '''绘制训练曲线'''
    def draw_train(self, history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

handler = SiameseNetwork()
handler.train_model()

