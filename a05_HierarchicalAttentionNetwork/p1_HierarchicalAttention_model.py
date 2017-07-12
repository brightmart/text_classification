# -*- coding: utf-8 -*-
# HierarchicalAttention: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier. 2017-06-13
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib

class HierarchicalAttention:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, num_sentences,
                 vocab_size, embed_size,
                 hidden_size, is_training, need_sentence_level_attention_encoder_flag=True, multi_label_flag=False,
                 initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0):#0.01
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_sentences = num_sentences
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")#TODO ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.multi_label_flag = multi_label_flag
        self.hidden_size = hidden_size
        self.need_sentence_level_attention_encoder_flag = need_sentence_level_attention_encoder_flag
        self.clip_gradients=clip_gradients

        # add placeholder (X,label)
        # self.input_x = tf.placeholder(tf.int32, [None, self.num_sentences,self.sequence_length], name="input_x")  # X
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")

        self.sequence_length = int(self.sequence_length / self.num_sentences) # TODO
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")  # y:[None,num_classes]
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes],name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.

        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]
        if not self.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),
                                          self.input_y)  # tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        else:
            self.accuracy = tf.constant(
                0.5)  # fuke accuracy. (you can calcuate accuracy outside of graph using method calculate_accuracy(...) in train.py)

        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()
        self.train_op = self.train()

    def attention_word_level(self, hidden_state):
        """
        input1:self.hidden_state: hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        input2:sentence level context vector:[batch_size*num_sentences,hidden_size*2]
        :return:representation.shape:[batch_size*num_sentences,hidden_size*2]
        """
        hidden_state_ = tf.stack(hidden_state, axis=1)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        # 0) one layer of feed forward network
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1,
                                                          self.hidden_size * 2])  # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        # hidden_state_:[batch_size*num_sentences*sequence_length,hidden_size*2];W_w_attention_sentence:[,hidden_size*2,,hidden_size*2]
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_word) + self.W_b_attention_word)  # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.sequence_length,
                                                                         self.hidden_size * 2])  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        # attention process:1.get logits for each word in the sentence. 2.get possibility distribution for each word in the sentence. 3.get weighted sum for the sentence as sentence representation.
        # 1) get logits for each word in the sentence.
        hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       self.context_vecotor_word)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # shape:[batch_size*num_sentences,sequence_length]
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
        attention_logits_max = tf.reduce_max(attention_logits, axis=1,
                                             keep_dims=True)  # shape:[batch_size*num_sentences,1]
        # 2) get possibility distribution for each word in the sentence.
        p_attention = tf.nn.softmax(
            attention_logits - attention_logits_max)  # shape:[batch_size*num_sentences,sequence_length]
        # 3) get weighted hidden state by attention vector
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # shape:[batch_size*num_sentences,sequence_length,1]
        # below sentence_representation'shape:[batch_size*num_sentences,sequence_length,hidden_size*2]<----p_attention_expanded:[batch_size*num_sentences,sequence_length,1];hidden_state_:[batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation,
                                                axis=1)  # shape:[batch_size*num_sentences,hidden_size*2]
        return sentence_representation  # shape:[batch_size*num_sentences,hidden_size*2]

    def attention_sentence_level(self, hidden_state_sentence):
        """
        input1: hidden_state_sentence: a list,len:num_sentence,element:[None,hidden_size*4]
        input2: sentence level context vector:[self.hidden_size*2]
        :return:representation.shape:[None,hidden_size*4]
        """
        hidden_state_ = tf.stack(hidden_state_sentence, axis=1)  # shape:[None,num_sentence,hidden_size*4]

        # 0) one layer of feed forward
        hidden_state_2 = tf.reshape(hidden_state_,
                                    shape=[-1, self.hidden_size * 4])  # [None*num_sentence,hidden_size*4]
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_sentence) + self.W_b_attention_sentence)  # shape:[None*num_sentence,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.num_sentences,
                                                                         self.hidden_size * 2])  # [None,num_sentence,hidden_size*2]
        # attention process:1.get logits for each sentence in the doc.2.get possibility distribution for each sentence in the doc.3.get weighted sum for the sentences as doc representation.
        # 1) get logits for each word in the sentence.
        hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       self.context_vecotor_sentence)  # shape:[None,num_sentence,hidden_size*2]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # shape:[None,num_sentence]. that is get logit for each num_sentence.
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:computes the maximum of elements across dimensions of a tensor.
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)  # shape:[None,1]
        # 2) get possibility distribution for each word in the sentence.
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)  # shape:[None,num_sentence]
        # 3) get weighted hidden state by attention vector(sentence level)
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # shape:[None,num_sentence,1]
        sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_)  # shape:[None,num_sentence,hidden_size*2]<---p_attention_expanded:[None,num_sentence,1];hidden_state_:[None,num_sentence,hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)  # shape:[None,hidden_size*2]
        return sentence_representation  # shape:[None,hidden_size*2]

    def inference(self):
        """main computation graph here: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier"""
        # 1.Word Encoder
        # 1.1 embedding of words
        input_x = tf.split(self.input_x, self.num_sentences,axis=1)  # a list. length:num_sentences.each element is:[None,self.sequence_length/num_sentences]
        input_x = tf.stack(input_x, axis=1)  # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,input_x)  # [None,num_sentences,sentence_length,embed_size]
        embedded_words_reshaped = tf.reshape(self.embedded_words, shape=[-1, self.sequence_length,self.embed_size])  # [batch_size*num_sentences,sentence_length,embed_size]
        # 1.2 forward gru
        hidden_state_forward_list = self.gru_forward_word_level(embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.3 backward gru
        hidden_state_backward_list = self.gru_backward_word_level(embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.4 concat forward hidden state and backward hidden state. hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        self.hidden_state = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                             zip(hidden_state_forward_list, hidden_state_backward_list)]  # hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]

        # 2.Word Attention
        # for each sentence.
        sentence_representation = self.attention_word_level(self.hidden_state)  # output:[batch_size*num_sentences,hidden_size*2]
        sentence_representation = tf.reshape(sentence_representation, shape=[-1, self.num_sentences, self.hidden_size * 2])  # shape:[batch_size,num_sentences,hidden_size*2]
        #with tf.name_scope("dropout"):#TODO
        #    sentence_representation = tf.nn.dropout(sentence_representation,keep_prob=self.dropout_keep_prob)  # shape:[None,hidden_size*4]

        # 3.Sentence Encoder
        # 3.1) forward gru for sentence
        hidden_state_forward_sentences = self.gru_forward_sentence_level(sentence_representation)  # a list.length is sentence_length, each element is [None,hidden_size]
        # 3.2) backward gru for sentence
        hidden_state_backward_sentences = self.gru_backward_sentence_level(sentence_representation)  # a list,length is sentence_length, each element is [None,hidden_size]
        # 3.3) concat forward hidden state and backward hidden state
        # below hidden_state_sentence is a list,len:sentence_length,element:[None,hidden_size*2]
        self.hidden_state_sentence = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in zip(hidden_state_forward_sentences, hidden_state_backward_sentences)]

        # 4.Sentence Attention
        document_representation = self.attention_sentence_level(self.hidden_state_sentence)  # shape:[None,hidden_size*4]
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(document_representation,keep_prob=self.dropout_keep_prob)  # shape:[None,hidden_size*4]
        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection  # shape:[None,self.num_classes]==tf.matmul([None,hidden_size*2],[hidden_size*2,self.num_classes])
        return logits

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.logits);  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss_multilabel(self, l2_lambda=0.00001*10): #*3#0.00001 #TODO 0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                             logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            loss = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def gru_single_step_word_level(self, Xt, h_t_minus_1):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size*num_sentences,embed_size]
        :param h_t_minus_1:[batch_size*num_sentences,embed_size]
        :return:
        """
        # update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1,
                                                                self.U_z) + self.b_z)  # z_t:[batch_size*num_sentences,self.hidden_size]
        # reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1,
                                                                self.U_r) + self.b_r)  # r_t:[batch_size*num_sentences,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) +r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)  # h_t_candiate:[batch_size*num_sentences,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t

    def gru_single_step_sentence_level(self, Xt,
                                       h_t_minus_1):  # Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
        """
        single step of gru for sentence level
        :param Xt:[batch_size, hidden_size*2]
        :param h_t_minus_1:[batch_size, hidden_size*2]
        :return:h_t:[batch_size,hidden_size]
        """
        # update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_sentence) + tf.matmul(h_t_minus_1,
                                                                         self.U_z_sentence) + self.b_z_sentence)  # z_t:[batch_size,self.hidden_size]
        # reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_sentence) + tf.matmul(h_t_minus_1,
                                                                         self.U_r_sentence) + self.b_r_sentence)  # r_t:[batch_size,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_sentence) + r_t * (
        tf.matmul(h_t_minus_1, self.U_h_sentence)) + self.b_h_sentence)  # h_t_candiate:[batch_size,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t

    # forward gru for first level: word levels
    def gru_forward_word_level(self, embedded_words):
        """
        :param embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return:forward hidden state: a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length,
                                           axis=1)  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,embed_size]
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in
                                  embedded_words_splitted]  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        # demension_1=embedded_words_squeeze[0].get_shape().dims[0]
        h_t = tf.ones((self.batch_size * self.num_sentences,
                       self.hidden_size))  #TODO self.hidden_size h_t =int(tf.get_shape(embedded_words_squeeze[0])[0]) # tf.ones([self.batch_size*self.num_sentences, self.hidden_size]) # [batch_size*num_sentences,embed_size]
        h_t_forward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):  # Xt: [batch_size*num_sentences,embed_size]
            h_t = self.gru_single_step_word_level(Xt,h_t)  # [batch_size*num_sentences,embed_size]<------Xt:[batch_size*num_sentences,embed_size];h_t:[batch_size*num_sentences,embed_size]
            h_t_forward_list.append(h_t)
        return h_t_forward_list  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]

    # backward gru for first level: word level
    def gru_backward_word_level(self, embedded_words):
        """
        :param   embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return: backward hidden state:a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length,
                                           axis=1)  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,embed_size]
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in
                                  embedded_words_splitted]  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        embedded_words_squeeze.reverse()  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        # demension_1=int(tf.get_shape(embedded_words_squeeze[0])[0]) #h_t = tf.ones([self.batch_size*self.num_sentences, self.hidden_size])
        h_t = tf.ones((self.batch_size * self.num_sentences, self.hidden_size))
        h_t_backward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_backward_list.append(h_t)
        h_t_backward_list.reverse() #ADD 2017.06.14
        return h_t_backward_list

    # forward gru for second level: sentence level
    def gru_forward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [batch_size,hidden_size]
        """
        # split embedded_words
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences,
                                                    axis=1)  # it is a list.length is num_sentences,each element is [batch_size,1,hidden_size*2]
        sentence_representation_squeeze = [tf.squeeze(x, axis=1) for x in
                                           sentence_representation_splitted]  # it is a list.length is num_sentences,each element is [batch_size, hidden_size*2]
        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0]) #scalar: batch_size
        h_t = tf.ones((self.batch_size, self.hidden_size * 2))  # TODO
        h_t_forward_list = []
        for time_step, Xt in enumerate(sentence_representation_squeeze):  # Xt:[batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level(Xt,
                                                      h_t)  # h_t:[batch_size,hidden_size]<---------Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
            h_t_forward_list.append(h_t)
        return h_t_forward_list  # a list,length is num_sentences, each element is [batch_size,hidden_size]

    # backward gru for second level: sentence level
    def gru_backward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [batch_size,hidden_size]
        """
        # split embedded_words
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences,
                                                    axis=1)  # it is a list.length is num_sentences,each element is [batch_size,1,hidden_size*2]
        sentence_representation_squeeze = [tf.squeeze(x, axis=1) for x in
                                           sentence_representation_splitted]  # it is a list.length is num_sentences,each element is [batch_size, hidden_size*2]
        sentence_representation_squeeze.reverse()
        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0])  # scalar: batch_size
        h_t = tf.ones((self.batch_size, self.hidden_size * 2))
        h_t_forward_list = []
        for time_step, Xt in enumerate(sentence_representation_squeeze):  # Xt:[batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level(Xt,h_t)  # h_t:[batch_size,hidden_size]<---------Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
            h_t_forward_list.append(h_t)
        h_t_forward_list.reverse() #ADD 2017.06.14
        return h_t_forward_list  # a list,length is num_sentences, each element is [batch_size,hidden_size]

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding_projection"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 4, self.num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])  #TODO [label_size]

        # GRU parameters:update gate related
        with tf.name_scope("gru_weights_word_level"):
            self.W_z = tf.get_variable("W_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_z = tf.get_variable("U_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
            # GRU parameters:reset gate related
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_r = tf.get_variable("U_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])

            self.W_h = tf.get_variable("W_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_h = tf.get_variable("U_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])

        with tf.name_scope("gru_weights_sentence_level"):
            self.W_z_sentence = tf.get_variable("W_z_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.U_z_sentence = tf.get_variable("U_z_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.b_z_sentence = tf.get_variable("b_z_sentence", shape=[self.hidden_size * 2])
            # GRU parameters:reset gate related
            self.W_r_sentence = tf.get_variable("W_r_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.U_r_sentence = tf.get_variable("U_r_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.b_r_sentence = tf.get_variable("b_r_sentence", shape=[self.hidden_size * 2])

            self.W_h_sentence = tf.get_variable("W_h_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.U_h_sentence = tf.get_variable("U_h_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.b_h_sentence = tf.get_variable("b_h_sentence", shape=[self.hidden_size * 2])

        with tf.name_scope("attention"):
            self.W_w_attention_word = tf.get_variable("W_w_attention_word",
                                                      shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                      initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[self.hidden_size * 2])

            self.W_w_attention_sentence = tf.get_variable("W_w_attention_sentence",
                                                          shape=[self.hidden_size * 4, self.hidden_size * 2],
                                                          initializer=self.initializer)
            self.W_b_attention_sentence = tf.get_variable("W_b_attention_sentence", shape=[self.hidden_size * 2])
            self.context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2],
                                                        initializer=self.initializer)  # TODO o.k to use batch_size in first demension?
            self.context_vecotor_sentence = tf.get_variable("what_is_the_informative_sentence",
                                                            shape=[self.hidden_size * 2], initializer=self.initializer)


# test started
def test():
    # below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes = 3
    learning_rate = 0.01
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 30
    num_sentences = 6  # number of sentences
    vocab_size = 10000
    embed_size = 100 #100
    hidden_size = 100
    is_training = True
    dropout_keep_prob = 1  # 0.5 #num_sentences
    textRNN = HierarchicalAttention(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                                    num_sentences, vocab_size, embed_size,
                                    hidden_size, is_training,multi_label_flag=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            # input_x should be:[batch_size, num_sentences,self.sequence_length]
            input_x = np.zeros((batch_size, sequence_length)) #num_sentences
            input_x[input_x > 0.5] = 1
            input_x[input_x <= 0.5] = 0
            input_y = np.array(
                [1, 0, 1, 1, 1, 2, 1, 1])  # np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
            loss, acc, predict, W_projection_value, _ = sess.run(
                [textRNN.loss_val, textRNN.accuracy, textRNN.predictions, textRNN.W_projection, textRNN.train_op],
                feed_dict={textRNN.input_x: input_x, textRNN.input_y: input_y,
                           textRNN.dropout_keep_prob: dropout_keep_prob})
            print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)
            # print("W_projection_value_:",W_projection_value)
#test()