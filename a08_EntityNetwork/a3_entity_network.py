# -*- coding: utf-8 -*-
# EntityNet:1.input encoder  2. dynamic emeory 3.output layer
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
import numpy as np
from tensorflow.contrib import rnn
#from  a07_Transformer.a2_multi_head_attention import MultiHeadAttention

class EntityNetwork:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, story_length,
                 vocab_size, embed_size,hidden_size, is_training, multi_label_flag=False,block_size=20,
                 initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0,use_bi_lstm=False,use_additive_attention=False):#0.01
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")#TODO ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.multi_label_flag = multi_label_flag
        self.hidden_size = hidden_size
        self.clip_gradients=clip_gradients
        self.story_length=story_length
        self.block_size=block_size
        self.use_bi_lstm=use_bi_lstm
        self.dimension=self.hidden_size*2 if self.use_bi_lstm else self.hidden_size #if use bi-lstm, set dimension value, so it can be used later for parameter.
        self.use_additive_attention=use_additive_attention

        # add placeholder (X,label)
        # self.input_x = tf.placeholder(tf.int32, [None, self.num_sentences,self.sequence_length], name="input_x")  # X
        self.story=tf.placeholder(tf.int32,[None,self.story_length,self.sequence_length],name="story")
        self.query = tf.placeholder(tf.int32, [None, self.sequence_length], name="question")

        self.answer_single = tf.placeholder(tf.int32, [None,], name="input_y")  # y:[None,num_classes]
        self.answer_multilabel = tf.placeholder(tf.float32, [None, self.num_classes],name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.

        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]
        if not self.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),self.answer_single)  # tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        else:
            self.accuracy = tf.constant(0.5)  # fuke accuracy. (you can calcuate accuracy outside of graph using method calculate_accuracy(...) in train.py)

        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()
        self.train_op = self.train()

    def inference(self):
        """main computation graph here: 1.input encoder 2.dynamic emeory 3.output layer """
        # 1.input encoder
        self.embedding_with_mask()
        if self.use_bi_lstm:
            self.input_encoder_bi_lstm()
        else:
            self.input_encoder_bow()
        # 2. dynamic emeory
        self.hidden_state=self.rnn_story() #[batch_size,block_size,hidden_size]. get hidden state after process the story

        # 3.output layer
        logits=self.output_module() #[batch_size,vocab_size]
        return logits

    def embedding_with_mask(self):
        # 1.1 embedding for story and query
        story_embedding = tf.nn.embedding_lookup(self.Embedding,self.story)  # [batch_size,story_length,sequence_length,embed_size]
        query_embedding=tf.nn.embedding_lookup(self.Embedding,self.query)    # [batch_size,sequence_length,embed_size]
        # 1.2 mask for story and query
        story_mask=tf.get_variable("story_mask",[self.sequence_length,1],initializer=tf.constant_initializer(1.0))
        query_mask=tf.get_variable("query_mask",[self.sequence_length,1],initializer=tf.constant_initializer(1.0))
        # 1.3 multiply of embedding and mask for story and query
        self.story_embedding=tf.multiply(story_embedding,story_mask)  # [batch_size,story_length,sequence_length,embed_size]
        self.query_embedding=tf.multiply(query_embedding,query_mask)  # [batch_size,sequence_length,embed_size]

    def input_encoder_bow(self):
        # 1.4 use bag of words to encoder story and query
        self.story_embedding=tf.reduce_sum(self.story_embedding,axis=2) #[batch_size,story_length,embed_size]
        self.query_embedding=tf.reduce_sum(self.query_embedding,axis=1)  #[batch_size,embed_size]

    def input_encoder_bi_lstm(self):
        """use bi-directional lstm to encode query_embedding:[batch_size,sequence_length,embed_size]
                                         and story_embedding:[batch_size,story_length,sequence_length,embed_size]
        output:query_embedding:[batch_size,hidden_size*2]  story_embedding:[batch_size,self.story_length,self.hidden_size*2]
        """
        #1. encode query: bi-lstm layer
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)  # forward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)  # backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell == rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
        query_hidden_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.query_embedding,dtype=tf.float32,scope="query_rnn")  # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        query_hidden_output = tf.concat(query_hidden_output, axis=2) #[batch_size,sequence_length,hidden_size*2]
        self.query_embedding=tf.reduce_sum(query_hidden_output,axis=1) #[batch_size,hidden_size*2]
        print("input_encoder_bi_lstm.self.query_embedding:",self.query_embedding)

        #2. encode story
        # self.story_embedding:[batch_size,story_length,sequence_length,embed_size]
        self.story_embedding=tf.reshape(self.story_embedding,shape=(-1,self.story_length*self.sequence_length,self.embed_size)) #[self.story_length*self.sequence_length,self.embed_size]
        lstm_fw_cell_story = rnn.BasicLSTMCell(self.hidden_size)  # forward direction cell
        lstm_bw_cell_story = rnn.BasicLSTMCell(self.hidden_size)  # backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell_story = rnn.DropoutWrapper(lstm_fw_cell_story, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell_story == rnn.DropoutWrapper(lstm_bw_cell_story, output_keep_prob=self.dropout_keep_prob)
        story_hidden_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_story, lstm_bw_cell_story, self.story_embedding,dtype=tf.float32,scope="story_rnn")
        story_hidden_output=tf.concat(story_hidden_output,axis=2) #[batch_size,story_length*sequence_length,hidden_size*2]
        story_hidden_output=tf.reshape(story_hidden_output,shape=(-1,self.story_length,self.sequence_length,self.hidden_size*2))
        self.story_embedding = tf.reduce_sum(story_hidden_output, axis=2)  # [batch_size,self.story_length,self.hidden_size*2]

    def activation(self,features, scope=None):  # scope=None
        with tf.variable_scope(scope, 'PReLU', initializer=self.initializer):
            alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
            pos = tf.nn.relu(features)
            neg = alpha * (features - tf.abs(features)) * 0.5
            return pos + neg

    def output_module(self):
        """
        1.use attention mechanism between query and hidden states, to get weighted sum of hidden state. 2.non-linearity of query and hidden state to get label.
        input: query_embedding:[batch_size,embed_size], hidden state:[batch_size,block_size,hidden_size] of memory
        :return:y: predicted label.[]
        """
        # 1.use attention mechanism between query and hidden states, to get weighted sum of hidden state.
        # 1.1 get possibility distribution (of similiarity)
        p=tf.nn.softmax(tf.multiply(tf.expand_dims(self.query_embedding,axis=1),self.hidden_state)) #shape:[batch_size,block_size,hidden_size]<---query_embedding_expand:[batch_size,1,hidden_size]; hidden_state:[batch_size,block_size,hidden_size]
        # 1.2 get weighted sum of hidden state
        u=tf.reduce_sum(tf.multiply(p,self.hidden_state),axis=1) #shape:[batch_size,hidden_size]<----------([batch_size,block_size,hidden_size],[batch_size,block_size,hidden_size])

        # 2.non-linearity of query and hidden state to get label
        H_u_matmul=tf.matmul(u,self.H)+self.h_u_bias #shape:[batch_size,hidden_size]<----([batch_size,hidden_size],[hidden_size,hidden_size])
        activation=self.activation(self.query_embedding + H_u_matmul,scope="query_add_hidden")           #shape:[batch_size,hidden_size]
        activation = tf.nn.dropout(activation,keep_prob=self.dropout_keep_prob) #shape:[batch_size,hidden_size]
        y=tf.matmul(activation,self.R)+self.y_bias #shape:[batch_size,vocab_size]<-----([batch_size,hidden_size],[hidden_size,vocab_size])
        return y #shape:[batch_size,vocab_size]

    def rnn_story(self):
        """
        run rnn for story to get last hidden state
        input is:  story:                 [batch_size,story_length,embed_size]
        :return:   last hidden state.     [batch_size,embed_size]
        """
        # 1.split input to get lists.
        input_split=tf.split(self.story_embedding,self.story_length,axis=1) #a list.length is:story_length.each element is:[batch_size,1,embed_size]
        input_list=[tf.squeeze(x,axis=1) for x in input_split]           #a list.length is:story_length.each element is:[batch_size,embed_size]
        # 2.init keys(w_all) and values(h_all) of memory
        h_all=tf.get_variable("hidden_states",shape=[self.block_size,self.dimension],initializer=self.initializer)# [block_size,hidden_size]
        w_all=tf.get_variable("keys",          shape=[self.block_size,self.dimension],initializer=self.initializer)# [block_size,hidden_size]
        # 3.expand keys and values to prepare operation of rnn
        w_all_expand=tf.tile(tf.expand_dims(w_all,axis=0),[self.batch_size,1,1]) #[batch_size,block_size,hidden_size]
        h_all_expand=tf.tile(tf.expand_dims(h_all,axis=0),[self.batch_size,1,1]) #[batch_size,block_size,hidden_size]
        # 4. run rnn using input with cell.
        for i,input in enumerate(input_list):
            h_all_expand=self.cell(input,h_all_expand,w_all_expand,i) #w_all:[batch_size,block_size,hidden_size]; h_all:[batch_size,block_size,hidden_size]
        return h_all_expand #[batch_size,block_size,hidden_size]

    def cell(self,s_t,h_all,w_all,i):
        """
        parallel implementation of single time step for compute of input with memory
        :param s_t:   [batch_size,hidden_size].vector representation of current input(is a sentence).notice:hidden_size=embedding_size
        :param w_all: [batch_size,block_size,hidden_size]
        :param h_all: [batch_size,block_size,hidden_size]
        :return: new hidden state: [batch_size,block_size,hidden_size]
        """
        # 1.gate
        s_t_expand=tf.expand_dims(s_t, axis=1)       #[batch_size,1,hidden_size]
        g=tf.nn.sigmoid(tf.multiply(s_t_expand,h_all)+tf.multiply(s_t_expand,w_all))#shape:[batch_size,block_size,hidden_size]

        # 2.candidate hidden state
        #below' shape:[batch_size*block_size,hidden_size]
        h_candidate_part1=tf.matmul(tf.reshape(h_all,shape=(-1,self.dimension)), self.U) + tf.matmul(tf.reshape(w_all,shape=(-1,self.dimension)), self.V)+self.h_bias
        print("======>h_candidate_part1:",h_candidate_part1) #(160, 100)
        h_candidate_part1=tf.reshape(h_candidate_part1,shape=(self.batch_size,self.block_size,self.dimension)) #[batch_size,block_size,hidden_size]
        h_candidate_part2=tf.expand_dims(tf.matmul(s_t,self.W)+self.h2_bias,axis=1)              #shape:[batch_size,1,hidden_size]
        h_candidate=self.activation(h_candidate_part1+h_candidate_part2,scope="h_candidate"+str(i))   #shape:[batch_size,block_size,hidden_size]

        # 3.update hidden state
        h_all=h_all+tf.multiply(g,h_candidate) #shape:[batch_size,block_size,hidden_size]

        # 4.normalized hidden state
        h_all=tf.nn.l2_normalize(h_all,-1) #shape:[batch_size,block_size,hidden_size]
        return h_all  #shape:[batch_size,block_size,hidden_size]

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_single,logits=self.logits);  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('bias' not in v.name ) and ('alpha' not in v.name)]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss_multilabel(self, l2_lambda=0.0001): #this loss function is for multi-label classification
        with tf.name_scope("loss"):
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.answer_multilabel,logits=self.logits);  #[None,self.num_classes]. losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            #losses=self.smoothing_cross_entropy(self.logits,self.answer_multilabel,self.num_classes) #shape=(512,)
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            loss = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if('bias' not in v.name ) and ('alpha' not in v.name)]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def smoothing_cross_entropy(self,logits, labels, vocab_size, confidence=0.9): #confidence = 1.0 - label_smoothing. where label_smooth=0.1. from http://github.com/tensorflow/tensor2tensor
        """Cross entropy with label smoothing to limit over-confidence."""
        with tf.name_scope("smoothing_cross_entropy", [logits, labels]):
            # Low confidence is given to all non-true labels, uniformly.
            low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
            # Normalizing constant is the best cross-entropy value with soft targets.
            # We subtract it just for readability, makes no difference on learning.
            normalizing = -(confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) * low_confidence * tf.log(low_confidence + 1e-20))
            # Soft targets.
            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence)
            xentropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=soft_targets)
        return xentropy - normalizing

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    #:param s_t: vector representation of current input(is a sentence). shape:[batch_size,sequence_length,embed_size]
    #:param h: value(hidden state).shape:[hidden_size]
    #:param w: key.shape:[hidden_size]
    def instantiate_weights(self):
        """define all weights here"""
        with tf.variable_scope("output_module"):
            self.H=tf.get_variable("H",shape=[self.dimension,self.dimension],initializer=self.initializer)
            self.R = tf.get_variable("R", shape=[self.dimension, self.num_classes], initializer=self.initializer)
            self.y_bias=tf.get_variable("y_bias",shape=[self.num_classes])
            self.b_projected = tf.get_variable("b_projection", shape=[self.num_classes])
            self.h_u_bias=tf.get_variable("h_u_bias",shape=[self.dimension])

        with tf.variable_scope("dynamic_memory"):
            self.U=tf.get_variable("U",shape=[self.dimension,self.dimension],initializer=self.initializer)
            self.V=tf.get_variable("V",shape=[self.dimension,self.dimension],initializer=self.initializer)
            self.W=tf.get_variable("W",shape=[self.dimension,self.dimension],initializer=self.initializer)
            self.h_bias=tf.get_variable("h_bias",shape=[self.dimension])
            self.h2_bias = tf.get_variable("h2_bias", shape=[self.dimension])

        with tf.variable_scope("embedding_projection"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)
            self.Embedding_label = tf.get_variable("Embedding_label", shape=[self.num_classes, self.embed_size],dtype=tf.float32) #,initializer=self.initializer
            #self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 4, self.num_classes],initializer=self.initializer)  # [embed_size,label_size]
            #self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

        with tf.variable_scope("attention"):
            self.W_w_attention_word = tf.get_variable("W_w_attention_word",shape=[self.hidden_size * 2, self.hidden_size * 2],initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[self.hidden_size * 2])
            self.context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2],initializer=self.initializer)  # TODO o.k to use batch_size in first demension?

# test: learn to count. weight of query and story is different
def test():
    # below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes = 15
    learning_rate = 0.001
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 10
    vocab_size = 10000
    embed_size = 100
    hidden_size = 100
    is_training = True
    story_length = 3
    dropout_keep_prob = 1
    use_bi_lstm=False
    model = EntityNetwork(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                          story_length, vocab_size, embed_size, hidden_size, is_training,
                          multi_label_flag=False, block_size=20,use_bi_lstm=use_bi_lstm)
    ckpt_dir = 'checkpoint_entity_network/dummy_test/'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1500):
            # input_x should be:[batch_size, num_sentences,self.sequence_length]
            story = np.random.randn(batch_size, story_length, sequence_length)
            story[story > 0] = 1
            story[story <= 0] = 0
            query = np.random.randn(batch_size, sequence_length)  # [batch_size, sequence_length]
            query[query > 0] = 1
            query[query <= 0] = 0
            answer_single = np.sum(query, axis=1) + np.round(0.1 * np.sum(np.sum(story, axis=1),
                                                                          axis=1))  # [batch_size].e.g. np.array([1, 0, 1, 1, 1, 2, 1, 1])
            loss, acc, predict, _ = sess.run(
                [model.loss_val, model.accuracy, model.predictions, model.train_op],
                feed_dict={model.query: query, model.story: story, model.answer_single: answer_single,
                           model.dropout_keep_prob: dropout_keep_prob})
            print(i, "query:", query, "=====================>")
            print(i, "loss:", loss, "acc:", acc, "label:", answer_single, "prediction:", predict)
            if i % 300 == 0:
                save_path = ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=i * 300)

def predict():
    num_classes = 15
    learning_rate = 0.001
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 10
    vocab_size = 10000
    embed_size = 100
    hidden_size = 100
    is_training = False
    story_length = 3
    dropout_keep_prob = 1
    model = EntityNetwork(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                          story_length, vocab_size, embed_size, hidden_size, is_training,
                          multi_label_flag=False, block_size=20)
    ckpt_dir = 'checkpoint_entity_network/dummy_test/'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        for i in range(1500):
            story = np.random.randn(batch_size, story_length, sequence_length)
            story[story > 0] = 1
            story[story <= 0] = 0
            query = np.random.randn(batch_size, sequence_length)  # [batch_size, sequence_length]
            query[query > 0] = 1
            query[query <= 0] = 0
            answer_single = np.sum(query, axis=1) + np.round(0.1 * np.sum(np.sum(story, axis=1),
                                                                          axis=1))  # [batch_size].e.g. np.array([1, 0, 1, 1, 1, 2, 1, 1])
            predict = sess.run([model.predictions], feed_dict={model.query: query, model.story: story,
                                                               model.dropout_keep_prob: dropout_keep_prob})
            print(i, "query:", query, "=====================>")
            print(i, "label:", answer_single, "prediction:", predict)

#test()
#predict()