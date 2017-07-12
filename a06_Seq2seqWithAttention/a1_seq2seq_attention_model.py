# -*- coding: utf-8 -*-
# seq2seq_attention: 1.word embedding 2.encoder 3.decoder(optional with attention). for more detail, please check:Neural Machine Translation By Jointly Learning to Align And Translate
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
import random
import copy
from a1_seq2seq import rnn_decoder_with_attention,extract_argmax_and_embed

class seq2seq_attention_model:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size,hidden_size, is_training,decoder_sent_length=6,
                 initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0,l2_lambda=0.0001):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.decoder_sent_length=decoder_sent_length
        self.hidden_size = hidden_size
        self.clip_gradients=clip_gradients
        self.l2_lambda=l2_lambda

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")                 #x
        self.decoder_input = tf.placeholder(tf.int32, [None, self.decoder_sent_length],name="decoder_input")  #y, but shift
        self.input_y_label = tf.placeholder(tf.int32, [None, self.decoder_sent_length], name="input_y_label") #y, but shift
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() #logits shape:[batch_size,decoder_sent_length,self.num_classes]

        self.predictions = tf.argmax(self.logits, axis=2, name="predictions")
        self.accuracy = tf.constant(0.5)  # fuke accuracy. (you can calcuate accuracy outside of graph using method calculate_accuracy(...) in train.py)
        if not is_training:
            return
        self.loss_val = self.loss_seq2seq()
        self.train_op = self.train()

    def inference(self):
        """main computation graph here:
        #1.Word embedding. 2.Encoder with GRU 3.Decoder using GRU(optional with attention)."""
        ###################################################################################################################################
        # 1.embedding of words
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)  #[None, self.sequence_length, self.embed_size]
        # 2.encoder with GRU
        # 2.1 forward gru
        hidden_state_forward_list = self.gru_forward(self.embedded_words,self.gru_cell)  # a list,length is sentence_length, each element is [batch_size,hidden_size]
        # 2.2 backward gru
        hidden_state_backward_list = self.gru_forward(self.embedded_words,self.gru_cell,reverse=True)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 2.3 concat forward hidden state and backward hidden state. hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        thought_vector_list=[tf.concat([h_forward,h_backward],axis=1) for h_forward,h_backward in zip(hidden_state_forward_list,hidden_state_backward_list)]#list,len:sent_len,e:[batch_size,hidden_size*2]

        # 3.Decoder using GRU with attention
        thought_vector=tf.stack(thought_vector_list,axis=1) #shape:[batch_size,sentence_length,hidden_size*2]
        #initial_state=tf.reduce_sum(thought_vector,axis=1) #[batch_size,hidden_size*2] #TODO NEED TO TEST WHICH ONE IS BETTER: SUM UP OR USE LAST HIDDEN STATE==>similiarity.
        initial_state=tf.nn.tanh(tf.matmul(hidden_state_backward_list[0],self.W_initial_state)+self.b_initial_state) #initial_state:[batch_size,hidden_size*2]. TODO this is follow paper's way.
        cell=self.gru_cell_decoder #this is a special cell. because it beside previous hidden state, current input, it also has a context vecotor, which represent attention result.

        output_projection=(self.W_projection,self.b_projection) #W_projection:[self.hidden_size * 2, self.num_classes]; b_projection:[self.num_classes]
        loop_function = extract_argmax_and_embed(self.Embedding_label,output_projection) if not self.is_training else None #loop function will be used only at testing, not training.
        attention_states=thought_vector #[None, self.sequence_length, self.embed_size]
        decoder_input_embedded=tf.nn.embedding_lookup(self.Embedding_label,self.decoder_input) #[batch_size,self.decoder_sent_length,embed_size]
        decoder_input_splitted = tf.split(decoder_input_embedded, self.decoder_sent_length,axis=1)  # it is a list,length is decoder_sent_length, each element is [batch_size,1,embed_size]
        decoder_input_squeezed = [tf.squeeze(x, axis=1) for x in decoder_input_splitted]  # it is a list,length is decoder_sent_length, each element is [batch_size,embed_size]

        #rnn_decoder_with_attention(decoder_inputs, initial_state, cell, loop_function,attention_states,scope=None):
            #input1:decoder_inputs:target, shift by one. for example.the target is:"X Y Z",then decoder_inputs should be:"START X Y Z" A list of 2D Tensors [batch_size x input_size].
            #input2:initial_state: 2D Tensor with shape  [batch_size x cell.state_size].
            #input3:attention_states:represent X. 3D Tensor [batch_size x attn_length x attn_size].
            #output:?
        outputs, final_state=rnn_decoder_with_attention(decoder_input_squeezed, initial_state, cell, loop_function, attention_states, scope=None) # A list.length:decoder_sent_length.each element is:[batch_size x output_size]
        decoder_output=tf.stack(outputs,axis=1) #decoder_output:[batch_size,decoder_sent_length,hidden_size*2]
        decoder_output=tf.reshape(decoder_output,shape=(-1,self.hidden_size*2)) #decoder_output:[batch_size*decoder_sent_length,hidden_size*2]

        with tf.name_scope("dropout"):
            decoder_output = tf.nn.dropout(decoder_output,keep_prob=self.dropout_keep_prob)  # shape:[None,hidden_size*4]
        # 4. get logits
        with tf.name_scope("output"):
            logits = tf.matmul(decoder_output, self.W_projection) + self.b_projection  # logits shape:[batch_size*decoder_sent_length,self.num_classes]==tf.matmul([batch_size*decoder_sent_length,hidden_size*2],[hidden_size*2,self.num_classes])
            logits=tf.reshape(logits,shape=(self.batch_size,self.decoder_sent_length,self.num_classes)) #logits shape:[batch_size,decoder_sent_length,self.num_classes]
        ###################################################################################################################################
        return logits

    def loss_seq2seq(self):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y_label, logits=self.logits);#losses:[batch_size,self.decoder_sent_length]
            loss_batch=tf.reduce_sum(losses,axis=1)/self.decoder_sent_length #loss_batch:[batch_size]
            loss=tf.reduce_mean(loss_batch)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            loss = loss + l2_losses
            return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def gru_cell(self, Xt, h_t_minus_1):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size,embed_size]
        :param h_t_minus_1:[batch_size,embed_size]
        :return:
        """
        # 1.update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1,self.U_z) + self.b_z)  # z_t:[batch_size,self.hidden_size]
        # 2.reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1,self.U_r) + self.b_r)  # r_t:[batch_size,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) +r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)  # h_t_candiate:[batch_size,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t

    def gru_cell_decoder(self, Xt, h_t_minus_1,context_vector):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size,embed_size]
        :param h_t_minus_1:[batch_size,embed_size]
        :param context_vector. [batch_size,embed_size].this represent the result from attention( weighted sum of input during current decoding step)
        :return:
        """
        # 1.update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_decoder) + tf.matmul(h_t_minus_1,self.U_z_decoder) +tf.matmul(context_vector,self.C_z_decoder)+self.b_z_decoder)  # z_t:[batch_size,self.hidden_size]
        # 2.reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_decoder) + tf.matmul(h_t_minus_1,self.U_r_decoder) +tf.matmul(context_vector,self.C_r_decoder)+self.b_r_decoder)  # r_t:[batch_size,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_decoder) +r_t * (tf.matmul(h_t_minus_1, self.U_h_decoder)) +tf.matmul(context_vector, self.C_h_decoder)+ self.b_h_decoder)  # h_t_candiate:[batch_size,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t,h_t

    # forward gru for first level: word levels
    def gru_forward(self, embedded_words,gru_cell, reverse=False):
        """
        :param embedded_words:[None,sequence_length, self.embed_size]
        :return:forward hidden state: a list.length is sentence_length, each element is [batch_size,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length,axis=1)  # it is a list,length is sentence_length, each element is [batch_size,1,embed_size]
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]  # it is a list,length is sentence_length, each element is [batch_size,embed_size]
        h_t = tf.ones((self.batch_size,self.hidden_size))
        h_t_list = []
        if reverse:
            embedded_words_squeeze.reverse()
        for time_step, Xt in enumerate(embedded_words_squeeze):  # Xt: [batch_size,embed_size]
            h_t = gru_cell(Xt,h_t) #h_t:[batch_size,embed_size]<------Xt:[batch_size,embed_size];h_t:[batch_size,embed_size]
            h_t_list.append(h_t)
        if reverse:
            h_t_list.reverse()
        return h_t_list  # a list,length is sentence_length, each element is [batch_size,hidden_size]

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("decoder_init_state"):
            self.W_initial_state = tf.get_variable("W_initial_state", shape=[self.hidden_size, self.hidden_size*2], initializer=self.initializer)
            self.b_initial_state = tf.get_variable("b_initial_state", shape=[self.hidden_size*2])
        with tf.name_scope("embedding_projection"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding_label = tf.get_variable("Embedding_label", shape=[self.num_classes, self.embed_size*2],dtype=tf.float32) #,initializer=self.initializer
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size*2, self.num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

        # GRU parameters:update gate related
        with tf.name_scope("gru_weights_encoder"):
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

        with tf.name_scope("gru_weights_decoder"):
            self.W_z_decoder = tf.get_variable("W_z_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.U_z_decoder = tf.get_variable("U_z_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.C_z_decoder = tf.get_variable("C_z_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],initializer=self.initializer) #TODO
            self.b_z_decoder = tf.get_variable("b_z_decoder", shape=[self.hidden_size*2])
            # GRU parameters:reset gate related
            self.W_r_decoder = tf.get_variable("W_r_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.U_r_decoder = tf.get_variable("U_r_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.C_r_decoder = tf.get_variable("C_r_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],initializer=self.initializer) #TODO
            self.b_r_decoder = tf.get_variable("b_r_decoder", shape=[self.hidden_size*2])

            self.W_h_decoder = tf.get_variable("W_h_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.U_h_decoder = tf.get_variable("U_h_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)   #TODO
            self.C_h_decoder = tf.get_variable("C_h_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],initializer=self.initializer)
            self.b_h_decoder = tf.get_variable("b_h_decoder", shape=[self.hidden_size*2])

        with tf.name_scope("full_connected"):
            self.W_fc=tf.get_variable("W_fc",shape=[self.hidden_size*2,self.hidden_size])
            self.a_fc=tf.get_variable("a_fc",shape=[self.hidden_size])

# test started: learn to output reverse sequence of itself.
def test():
    # below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes = 9+2 #additional two classes:one is for _GO, another is for _END
    learning_rate = 0.0001
    batch_size = 1
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 300
    embed_size = 100 #100
    hidden_size = 100
    is_training = True
    dropout_keep_prob = 1  # 0.5 #num_sentences
    decoder_sent_length=6
    l2_lambda=0.0001
    model = seq2seq_attention_model(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                                    vocab_size, embed_size,hidden_size, is_training,decoder_sent_length=decoder_sent_length,l2_lambda=l2_lambda)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1500):
            #input_x = np.zeros((batch_size, sequence_length),dtype=np.int32) #[None, self.sequence_length]
            label_list=get_unique_labels()
            input_x = np.array([label_list],dtype=np.int32) #[2,3,4,5,6]
            label_list_original=copy.deepcopy(label_list)
            label_list.reverse()
            decoder_input=np.array([[0]+label_list],dtype=np.int32) #[[0,2,3,4,5,6]]
            input_y_label=np.array([label_list+[1]],dtype=np.int32) #[[2,3,4,5,6,1]]
            loss, acc, predict, W_projection_value, _ = sess.run([model.loss_val, model.accuracy, model.predictions, model.W_projection, model.train_op],
                                                     feed_dict={model.input_x:input_x,model.decoder_input:decoder_input, model.input_y_label: input_y_label,
                                                                model.dropout_keep_prob: dropout_keep_prob})
            print(i,"loss:", loss, "acc:", acc, "label_list_original as input x:",label_list_original,";input_y_label:", input_y_label, "prediction:", predict)

def get_unique_labels():
    x=[2,3,4,5,6]
    random.shuffle(x)
    return x
test()