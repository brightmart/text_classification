# -*- coding: utf-8 -*-
#TextCNN: for each of two sentences,do(1. embeddding layers, 2.convolutional layer, 3.max-pooling,),4. concat features,5.FC, 6.softmax layer.
# print("started...")
import tensorflow as tf
import numpy as np
import random
from tflearn.data_utils import pad_sequences #to_categorical

class TwoCNNTextRelation:
    def __init__(self, filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,
                 embed_size,is_training,initializer=tf.random_normal_initializer(stddev=0.1)):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.is_training=is_training
        self.learning_rate=learning_rate
        self.filter_sizes=filter_sizes # it is a list of int. e.g. [3,4,5]
        self.num_filters=num_filters
        self.initializer=initializer
        self.num_filters_total=self.num_filters * len(filter_sizes) #how many filters totally.

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X: first sentence
        self.input_x2 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x2")  # X: second sentence
        self.input_y = tf.placeholder(tf.int32, [None,],name="input_y")  # y: 0 or 1. 1 means two sentences related to each other;0 means no relation.
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.instantiate_weights()
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]

        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection",shape=[self.num_filters_total*2, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size]

    def inference(self):
        """main computation graph here: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer."""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words1 = tf.nn.embedding_lookup(self.Embedding,self.input_x)#[None,sentence_length,embed_size]
        self.sentence_embeddings_expanded1=tf.expand_dims(self.embedded_words1,-1) #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        self.embedded_words2 = tf.nn.embedding_lookup(self.Embedding,self.input_x2)#[None,sentence_length,embed_size]
        self.sentence_embeddings_expanded2=tf.expand_dims(self.embedded_words2,-1) #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        #2.1 get features of sentence1
        h1=self.conv_relu_pool_dropout(self.sentence_embeddings_expanded1,name_scope_prefix="s1") #[None,num_filters_total]
        #2.2 get features of sentence2
        h2 =self.conv_relu_pool_dropout(self.sentence_embeddings_expanded2,name_scope_prefix="s2")  # [None,num_filters_total]
        #3. concat features
        h=tf.concat([h1,h2],axis=1) #[None,num_filters_total*2]
        #4. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(h,self.W_projection) + self.b_projection  #shape:[None, self.num_classes]==tf.matmul([None,self.num_filters_total*2],[self.num_filters_total*2,self.num_classes])
        return logits

    def conv_relu_pool_dropout(self,sentence_embeddings_expanded, name_scope_prefix=None):
        # 1.loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope(name_scope_prefix + "convolution-pooling-%s" % filter_size):
                # ====>a.create filter
                filter = tf.get_variable(name_scope_prefix+"filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer)
                # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                # Conv.Returns: A `Tensor`. Has the same type as `input`.
                #         A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                # 1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                # input data format:NHWC:[batch, height, width, channels];output:4-D
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # ====>c. apply nolinearity
                b = tf.get_variable(name_scope_prefix+"b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b),
                               "relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                #                  ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                #                  strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool")  # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs.append(pooled)
        # 2.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        h_pool = tf.concat(pooled_outputs,
                           3)  # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        h_pool_flat = tf.reshape(h_pool, [-1,
                                          self.num_filters_total])  # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
        # 3.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]
        return h_drop

    def loss(self,l2_lambda=0.0001):#0.001
        with tf.name_scope("loss"):
            #input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);#sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def loss_multilabel(self,l2_lambda=0.001): #this loss function is for multi-label classification
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            #input_y:shape=(?, 1999); logits:shape=(?, 1999)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits);#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, 1999)
            losses=tf.reduce_sum(losses,axis=1) #shape=(?,)
            loss=tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss


    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op

#test started
# def test():
#     #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
#     filter_sizes=[3,4,5]
#     num_filters=128
#     num_classes=2
#     learning_rate=0.001
#     batch_size=1
#     decay_steps=1000
#     decay_rate=0.9
#     sequence_length=15
#     vocab_size=100
#     embed_size=100
#     is_training=True
#     dropout_keep_prob=1
#     twoCNNTR=TwoCNNTextRelation(filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,
#                                           sequence_length,vocab_size,embed_size,is_training)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for i in range(10000):
#             x1=random.choice([1,2,3,4,5,6,7,8,9])
#             x2 = random.choice([1,2,3,4,5,6,7,8,9])
#             #input_x=np.array([[x1,x1,x1,0,x2,x2,x2]])
#             label=0
#             if np.abs(x1-x2)<3:label=1
#             input_y=np.array([label]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
#             x1 = pad_sequences([[x1,x1,x1,x1,x1]], maxlen=sequence_length, value=0)
#             x2 = pad_sequences([[x2,x2,x2,x2,x2]], maxlen=sequence_length, value=0)
#             loss_sum=0.0
#             acc_sum=0.0
#             loss,acc,predict,_=sess.run([twoCNNTR.loss_val,twoCNNTR.accuracy,twoCNNTR.predictions,twoCNNTR.train_op],
#                                         feed_dict={twoCNNTR.input_x:x1,twoCNNTR.input_x2:x2,twoCNNTR.input_y:input_y,
#                                                    twoCNNTR.dropout_keep_prob:dropout_keep_prob})
#             loss_sum=loss_sum+loss
#             acc_sum=acc_sum+acc
#             if i%100==0:
#                 print(i,"x1:",x1,";x2:",x2,"loss:",loss,"acc:",acc,"avg loss:",loss_sum/(i+1),";avg acc:",acc_sum/(i+1),"label:",input_y,"prediction:",predict)
# test()