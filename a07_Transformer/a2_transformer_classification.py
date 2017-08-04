# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import copy
from a2_base_model import BaseClass
from a2_encoder import Encoder
import os
"""
Transformer_classification: originally it perform sequence to sequence solely on attention mechanism. do it fast and better. now we use it to do text classification.
for more detail, check paper: "Attention Is All You Need"
1. position embedding for encoder input and decoder input
2. encoder with multi-head attention, position-wise feed forward
3. decoder with multi-head attention for decoder input,position-wise feed forward, mulit-head attention between encoder and decoder.

encoder:
6 layers.each layers has two sub-layers.
the first is multi-head self-attention mechanism;
the second is position-wise fully connected feed-forward network.
for each sublayer. use LayerNorm(x+Sublayer(x)). all dimension=512.

Decoder:
1. The decoder is composed of a stack of N= 6 identical layers.
2. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
attention over the output of the encoder stack.
3. Similar to the encoder, we employ residual connections
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions.  This
masking, combined with fact that the output embeddings are offset by one position, ensures that the
predictions for position i can depend only on the known outputs at positions less than i.
"""
class Transformer(BaseClass):
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size,d_model,d_k,d_v,h,num_layer,is_training,
                 initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0,l2_lambda=0.0001,use_residual_conn=False):
        """init all hyperparameter here"""
        super(Transformer, self).__init__(d_model, d_k, d_v, sequence_length, h, batch_size, num_layer=num_layer) #init some fields by using parent class.

        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = d_model
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.clip_gradients=clip_gradients
        self.l2_lambda=l2_lambda

        self.is_training=is_training #self.is_training=tf.placeholder(tf.bool,name="is_training") #tf.bool #is_training
        self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name="input_x")                 #x  batch_size
        self.input_y_label = tf.placeholder(tf.int32, [self.batch_size], name="input_y_label")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.use_residual_conn=use_residual_conn

        self.instantiate_weights()
        self.logits = self.inference() #logits shape:[batch_size,self.num_classes]

        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),self.input_y_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        if self.is_training is False:# if it is not training, then no need to calculate loss and back-propagation.
            return
        self.loss_val = self.loss()
        self.train_op = self.train()

    def inference(self):
        """ building blocks:
        encoder:6 layers.each layers has two   sub-layers. the first is multi-head self-attention mechanism; the second is position-wise fully connected feed-forward network.
               for each sublayer. use LayerNorm(x+Sublayer(x)). all dimension=512.
        decoder:6 layers.each layers has three sub-layers. the second layer is performs multi-head attention over the ouput of the encoder stack.
               for each sublayer. use LayerNorm(x+Sublayer(x)).
        """
        # 1.embedding for encoder input & decoder input
        # 1.1 position embedding for encoder input
        input_x_embeded = tf.nn.embedding_lookup(self.Embedding,self.input_x)  #[None,sequence_length, embed_size]
        input_x_embeded=tf.multiply(input_x_embeded,tf.sqrt(tf.cast(self.d_model,dtype=tf.float32)))
        input_mask=tf.get_variable("input_mask",[self.sequence_length,1],initializer=self.initializer)
        input_x_embeded=tf.add(input_x_embeded,input_mask) #[None,sequence_length,embed_size].position embedding.

        # 2. encoder
        encoder_class=Encoder(self.d_model,self.d_k,self.d_v,self.sequence_length,self.h,self.batch_size,self.num_layer,input_x_embeded,input_x_embeded,dropout_keep_prob=self.dropout_keep_prob,use_residual_conn=self.use_residual_conn)
        Q_encoded,K_encoded = encoder_class.encoder_fn() #K_v_encoder

        Q_encoded=tf.reshape(Q_encoded,shape=(self.batch_size,-1)) #[batch_size,sequence_length*d_model]
        with tf.variable_scope("output"):
            logits = tf.matmul(Q_encoded, self.W_projection) + self.b_projection #logits shape:[batch_size*decoder_sent_length,self.num_classes]
        print("logits:",logits)
        return logits

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y_label,logits=self.logits);  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('bias' not in v.name ) and ('alpha' not in v.name)]) * l2_lambda
            loss = loss + l2_losses
        return loss

    #def loss_seq2seq(self):
    #    with tf.variable_scope("loss"):
    #        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y_label, logits=self.logits);#losses:[batch_size,self.decoder_sent_length]
    #        loss_batch=tf.reduce_sum(losses,axis=1)/self.decoder_sent_length #loss_batch:[batch_size]
    #        loss=tf.reduce_mean(loss_batch)
    #        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
    #        loss = loss + l2_losses
    #        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def instantiate_weights(self):
        """define all weights here"""
        with tf.variable_scope("embedding_projection"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding_label = tf.get_variable("Embedding_label", shape=[self.num_classes, self.embed_size],dtype=tf.float32) #,initializer=self.initializer
            self.W_projection = tf.get_variable("W_projection", shape=[self.sequence_length*self.d_model, self.num_classes],initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def get_mask(self,sequence_length):
        lower_triangle = tf.matrix_band_part(tf.ones([sequence_length, sequence_length]), -1, 0)
        result = -1e9 * (1.0 - lower_triangle)
        print("get_mask==>result:", result)
        return result
# test started: learn to predict the bigger number in two numbers from specific location of array.
def test_training():
    # below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes = 9+2 #additional two classes:one is for _GO, another is for _END
    learning_rate = 0.0001 #/10.0
    batch_size = 1
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 6 #5 TODO
    vocab_size = 300
    is_training = True #True
    dropout_keep_prob = 0.9  # 0.5 #num_sentences
    #decoder_sent_length=6
    l2_lambda=0.0001#0.0001
    d_model=512 #512
    d_k=64
    d_v=64
    h=8
    num_layer=1
    embed_size = d_model

    model = Transformer(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                        vocab_size, embed_size,d_model,d_k,d_v,h,num_layer,is_training,l2_lambda=l2_lambda)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt_dir = 'checkpoint_transformer/sequence_reverse/'
        if os.path.exists(ckpt_dir+"checkpoint"):
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        for i in range(15000):
            label_list=get_unique_labels()
            input_x = np.array([label_list+[9]],dtype=np.int32)
            label_list_original=copy.deepcopy(label_list)
            label_list.reverse()
            input_y_label=np.array([np.max([label_list[0],label_list[1]])],dtype=np.int32)

            loss, acc, predict, W_projection_value, _ = sess.run([model.loss_val, model.accuracy, model.predictions, model.W_projection, model.train_op],
                                                     feed_dict={model.input_x:input_x, model.input_y_label: input_y_label,
                                                                model.dropout_keep_prob: dropout_keep_prob}) #model.dropout_keep_prob: dropout_keep_prob
            print(i,"loss:", loss, "acc:", acc, "label_list_original as input x:",label_list_original,";input_y_label:", input_y_label, "prediction:", predict)
            if i%1500==0:
                save_path = ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=i)

def test_predict():
    # below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes = 9+2 #additional two classes:one is for _GO, another is for _END
    learning_rate = 0.001
    batch_size = 1
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 6 #5
    vocab_size = 300
    is_training = False #True
    dropout_keep_prob = 1  # 0.5 #num_sentences
    l2_lambda=0.0001
    d_model=512 #512
    d_k=64
    d_v=64
    h=8
    num_layer=1#6
    embed_size = d_model
    model = Transformer(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                                    vocab_size, embed_size,d_model,d_k,d_v,h,num_layer,is_training,l2_lambda=l2_lambda)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt_dir = 'checkpoint_transformer/sequence_reverse/'
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        print("=================restored.")
        for i in range(1500):
            label_list=get_unique_labels()
            input_x = np.array([label_list+[9]],dtype=np.int32)
            label_list_original=copy.deepcopy(label_list)
            label_list.reverse()
            input_y_label=np.array([np.max([label_list[0],label_list[1]])],dtype=np.int32)

            predict, W_projection_value = sess.run([ model.predictions, model.W_projection], #model.loss_val,--->loss, model.train_op
                                feed_dict={model.input_x:input_x,
                                           model.dropout_keep_prob: dropout_keep_prob})
            print(i, "label_list_original as input x:",label_list_original, "prediction:", predict,";label:",input_y_label) #"acc:", acc, "loss:", loss ";input_y_label:", input_y_label

def get_unique_labels(length=5):
    #if length is  None:
    #    x=[2,3,4,5,6]
    #else:
    x=[i for i in range(2,2+length)]
    random.shuffle(x)
    return x

def get_unique_labels_batch(batch_size,length=None):
    x=[]
    for i in range(batch_size):
        labels=get_unique_labels(length=length)
        x.append(labels)
    return x


#test_training()
#test_predict()
#test_training_batch()
#test_training()