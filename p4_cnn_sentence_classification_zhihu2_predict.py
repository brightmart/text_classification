# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

"""
Simple example using convolutional neural network to classify IMDB
sentiment dataset.
References:
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
    - Kim Y. Convolutional Neural Networks for Sentence Classification[C].
    Empirical Methods in Natural Language Processing, 2014.
Links:
    - http://ai.stanford.edu/~amaas/data/sentiment/
    - http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf
"""
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
#from tflearn.datasets import imdb
from p4_zhihu_load_data import load_data,create_voabulary,create_voabulary_label
import numpy as np
import pickle
import os
#import tflearn.metrics.Metric.Top_k as Top_k

print("started...")
f_cache='data_zhihu.pik'
# 1. loading dataset
trainX,trainY,testX,testY=None,None,None,None
number_classes=1999
#if os.path.exists(f_cache):
#    with open(f_cache, 'r') as f:
#        trainX,trainY,testX,testY,vocab_size=pickle.load(f)
#if trainX is None or trainY is None: #如果训练数据，不存在
#-------------------------------------------------------------------------------------------------
print("training data not exist==>load data, and dump it to file system")
vocabulary_word2index, vocabulary_index2word = create_voabulary()
vocab_size=len(vocabulary_word2index)
vocabulary_word2index_label = create_voabulary_label()
train, test, _ =load_data(vocabulary_word2index, vocabulary_word2index_label)
trainX, trainY = train
testX, testY = test
print("testX.shape:",np.array(testX).shape) #2500个list.每个list代表一句话
print("testY.shape:",np.array(testY).shape) #2500个label
print("testX[0]:",testX[0]) #[17, 25, 10, 406, 26, 14, 56, 61, 62, 323, 4]
print("testX[1]:",testX[1]);print("testY[0]:",testY[0]) #0 ;print("testY[1]:",testY[1]) #0

# 2.Data preprocessing
# Sequence padding
print("start padding & transform to one hot...")
trainX = pad_sequences(trainX, maxlen=100, value=0.) #padding to max length
testX = pad_sequences(testX, maxlen=100, value=0.)   #padding to max length
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=number_classes) #y as one hot
testY = to_categorical(testY, nb_classes=number_classes)   #y as one hot
print("end padding & transform to one hot...")
#--------------------------------------------------------------------------------------------------
    # cache trainX,trainY,testX,testY for next time use.
#    with open(f_cache, 'w') as f:
#        pickle.dump((trainX,trainY,testX,testY,vocab_size),f)
#else:
#    print("traning data exists in cache. going to use it.")

# 3.Building convolutional network
######################################MODEL:1.conv-2.conv-3.conv-4.max_pool-5.dropout-6.FC##############################################################################################
#(shape=None, placeholder=None, dtype=tf.float32,data_preprocessing=None, data_augmentation=None,name="InputData")
network = input_data(shape=[None, 100], name='input') #[None, 100] `input_data` is used as a data entry (placeholder) of a network. This placeholder will be feeded with data when training
network = tflearn.embedding(network, input_dim=vocab_size, output_dim=128) #TODO [None, 100,128].embedding layer for a sequence of ids. network: Incoming 2-D Tensor. input_dim: vocabulary size, oput_dim:embedding size
         #conv_1d(incoming,nb_filter,filter_size)
branch1 = conv_1d(network, 128, 1, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, 128, 2, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2") # [batch_size, new steps1, nb_filters]. padding:"VALID",only ever drops the right-most columns
branch4 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2") # [batch_size, new steps2, nb_filters]
branch5 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2") # [batch_size, new steps3, nb_filters]
network = merge([branch1, branch2, branch3,branch4,branch5], mode='concat', axis=1) # merge a list of `Tensor` into a single one.===>[batch_size, new steps1+new step2+new step3, nb_filters]
network = tf.expand_dims(network, 2) #[batch_size, new steps1+new step2+new step3,1, nb_filters] Inserts a dimension of 1 into a tensor's shape
network = global_max_pool(network) #[batch_size, pooled dim]
network = dropout(network, 0.5) #[batch_size, pooled dim]
network = fully_connected(network, number_classes, activation='softmax') #matmul([batch_size, pooled_dim],[pooled_dim,2])---->[batch_size,number_classes]
top5 = tflearn.metrics.Top_k(k=5)
network = regression(network, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy', name='target') #metric=top5
######################################MODEL:1.conv-2.conv-3.conv-4.max_pool-5.dropout-6.FC################################################################################################
# 4.Training
model = tflearn.DNN(network, tensorboard_verbose=0)
#model.fit(trainX, trainY, n_epoch = 10, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=256) #32
#model.save('model_zhihu_cnn12345')
model.load('model_zhihu_cnn12345')
print("going to make a prediction...")
predict_result=model.predict(testX[0:1000])
print("predict_result:",predict_result)
print("ended...")