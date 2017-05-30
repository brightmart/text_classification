# -*- coding: utf-8 -*-
"""
BiLstmTextRelation: check reationship of two questions(Qi,Qj),result(0 or 1). 1 means related,0 means no relation
input_x e.g. "how much is the computer? EOS price of laptop".two different questions were splitted by a special token: EOS
main graph:1. embeddding layer, 2.Bi-LSTM layer, 3.mean pooling, 4.FC layer, 5.softmax
"""
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random

class BiLstmTextRelation:
    def __init__(self,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,
                 vocab_size,embed_size,is_training,initializer=tf.random_normal_initializer(stddev=0.1)):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.hidden_size=embed_size
        self.is_training=is_training
        self.learning_rate=learning_rate
        self.initializer=initializer

        # add placeholder (X,label)
        #X:input_x e.g. "how much is the computer? EOS price of laptop"
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X:  concat of two sentence, split by EOS.
        self.input_y = tf.placeholder(tf.int32,[None], name="input_y")  # y [None,num_classes]
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
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()
    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*2, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size]

    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.mean pooling, 4.FC layer, 5.softmax """
        #1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x) #shape:[None,sentence_length,embed_size]
        #2. Bi-lstm layer
        # define lstm cess:get lstm cell output
        lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) #forward direction cell
        lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) #backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell=rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell==rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
        # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
        #                            output: A tuple (outputs, output_states)
        #                                    where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.embedded_words,dtype=tf.float32) #[batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        print("outputs:===>",outputs) #outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))
        #3. concat output
        output_rnn=tf.concat(outputs,axis=2) #[batch_size,sequence_length,hidden_size*2]
        output_rnn_pooled=tf.reduce_mean(output_rnn,axis=1) #[batch_size,hidden_size*2] #output_rnn_last=output_rnn[:,-1,:] ##[batch_size,hidden_size*2] #TODO
        print("output_rnn_pooled:", output_rnn_pooled) # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
        #4. logits(use linear layer)
        with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            logits = tf.matmul(output_rnn_pooled, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits

    def loss(self,l2_lambda=0.0001):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);#sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op


# def test():
#     #below is a function test; if you use this for text related task, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
#     #input_x e.g. "how much is the computer? EOS price of laptop"--->"1 2 3 4 5 6 0 11 12 13 14"; label:1
#     #input_x e.g. "how much is the computer? EOS what's the weather today?"--->"1 2 3 4 5 6 0 15 16 17 18 19"; label:0
#     num_classes=2
#     learning_rate=0.001
#     batch_size=1
#     decay_steps=1000
#     decay_rate=0.9
#     sequence_length=3
#     vocab_size=10000
#     embed_size=100
#     is_training=True
#     dropout_keep_prob=1#0.5
#     textRNN=BiLstmTextRelation(num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for i in range(10000):
#             x1=random.choice([1,2,3,4,5,6,7,8,9])
#             x2 = random.choice([1,2,3,4,5,6,7,8,9])
#             input_x=np.array([[x1,0,x2]]) #np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
#             label=0
#             if np.abs(x1-x2)<3:label=1
#             input_y=np.array([label]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
#             loss,acc,predict,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
#             if i%100==0:
#                 print(i,"x1:",x1,";x2:",x2,";loss:",loss,";acc:",acc,";label:",input_y,";prediction:",predict)
# test()