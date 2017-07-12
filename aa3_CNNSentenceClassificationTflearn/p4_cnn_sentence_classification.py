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
from tflearn.datasets import imdb
import numpy as np

print("started...")
# 1.IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,valid_portion=0.1)
trainX, trainY = train
testX, testY = test
print("testX.shape:",np.array(testX).shape) #2500个list.每个list代表一句话
print("testY.shape:",np.array(testY).shape) #2500个label
print("testX[0]:",testX[0]) #[17, 25, 10, 406, 26, 14, 56, 61, 62, 323, 4]
print("testY[0]:",testY[0]) #0

# 2.Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.) #padding to max length
testX = pad_sequences(testX, maxlen=100, value=0.)   #padding to max length
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2) #y as one hot
testY = to_categorical(testY, nb_classes=2)   #y as one hot

# 3.Building convolutional network
#(shape=None, placeholder=None, dtype=tf.float32,data_preprocessing=None, data_augmentation=None,name="InputData")
network = input_data(shape=[None, 100], name='input') #[None, 100] `input_data` is used as a data entry (placeholder) of a network. This placeholder will be feeded with data when training
network = tflearn.embedding(network, input_dim=10000, output_dim=128) #[None, 100,128].embedding layer for a sequence of ids. network: Incoming 2-D Tensor. input_dim: vocabulary size, oput_dim:embedding size
         #conv_1d(incoming,nb_filter,filter_size)
branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2") # [batch_size, new steps1, nb_filters]. padding:"VALID",only ever drops the right-most columns
branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2") # [batch_size, new steps2, nb_filters]
branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2") # [batch_size, new steps3, nb_filters]
network = merge([branch1, branch2, branch3], mode='concat', axis=1) # merge a list of `Tensor` into a single one.===>[batch_size, new steps1+new step2+new step3, nb_filters]
network = tf.expand_dims(network, 2) #[batch_size, new steps1+new step2+new step3,1, nb_filters] Inserts a dimension of 1 into a tensor's shape
network = global_max_pool(network) #[batch_size, pooled dim]
network = dropout(network, 0.5) #[batch_size, pooled dim]
network = fully_connected(network, 2, activation='softmax') #matmul([batch_size, pooled_dim],[pooled_dim,2])---->[batch_size,2]
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch = 5, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)
print("ended...")