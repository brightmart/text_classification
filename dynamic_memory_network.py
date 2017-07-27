# -*- coding: utf-8 -*-
"""
Dynamic Memory Network: a.Input Module,b.Question Module,c.Episodic Memory Module,d.Answer Module.
  1.Input Module: encode raw texts into vector representation
  2.Question Module: encode question into vector representation
  3.Episodic Memory Module: with inputs,it chooses which parts of inputs to focus on through the attention mechanism,
                            taking into account of question and previous memory====>it poduce a 'memory' vecotr.
  4.Answer Module:generate an answer from the final memory vector.
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
import numpy as np
from tensorflow.contrib import rnn

class DynamicMemoryNetwork:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, story_length,
                 vocab_size, embed_size,hidden_size, is_training, num_pass=2,use_gated_gru=False,sequence_decoder=False,multi_label_flag=False,block_size=20,
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
        self.num_pass=num_pass #number of pass to run for episodic memory module. for example, num_pass=2
        self.use_gated_gru=use_gated_gru #if this is True. we will use gated gru as our 'Memory Update Mechanism'
        self.sequence_decoder=sequence_decoder

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
        """main computation graph here: a.Input Module,b.Question Module,c.Episodic Memory Module,d.Answer Module """
        # 1.Input Module
        self.input_module() #[batch_size,story_length,hidden_size
        # 2.question module
        self.question_module() #[batch_size,hidden_size]
        # 3.episodic memory module
        self.episodic_memory_module() #[batch_size,hidden_size]
        # 4. answer module
        logits=self.answer_module() #[batch_size,vocab_size]
        return logits

    def input_module(self):
        """encode raw texts into vector representation"""
        story_embedding=tf.nn.embedding_lookup(self.Embedding,self.story)  # [batch_size,story_length,sequence_length,embed_size]
        story_embedding=tf.reshape(story_embedding,(self.batch_size,self.story_length,self.sequence_length*self.embed_size))
        hidden_state=tf.ones((self.batch_size,self.hidden_size),dtype=tf.float32)
        cell = rnn.GRUCell(self.hidden_size)
        self.story_embedding,hidden_state=tf.nn.dynamic_rnn(cell,story_embedding,dtype=tf.float32,scope="input_module")
        print("1.input_module.self.story_embedding:",self.story_embedding)

    def question_module(self):
        """
        input:tokens of query:[batch_size,sequence_length]
        :return: representation of question:[batch_size,hidden_size]
        """
        query_embedding = tf.nn.embedding_lookup(self.Embedding, self.query)  # [batch_size,sequence_length,embed_size]
        cell=rnn.GRUCell(self.hidden_size)
        _,self.query_embedding=tf.nn.dynamic_rnn(cell,query_embedding,dtype=tf.float32,scope="question_module") #query_embedding:[batch_size,hidden_size]
        print("2.input_module.self.query_embedding:", self.query_embedding)

    def episodic_memory_module(self):#input(story)
        """
        episodic memory module
        1.combine features
        1.attention mechansim using gate function.take fact representation c,question q,previous memory m_previous
        2.use gated-gru to update hidden state
        3.set last hidden state as episode result
        4.use gru to update final memory using episode result

        input: story(from input module):[batch_size,story_length,hidden_size]
        output: last hidden state:[batch_size,hidden_size]
        """
        #self.story_embedding. [batch_size,story_length,hidden_size]
        candidate_inputs=tf.split(self.story_embedding,self.story_length,axis=1) # a list. length is: story_length. each element is:[batch_size,1,embedding_size]
        candidate_list=[tf.squeeze(x,axis=1) for x in candidate_inputs]          # a list. length is: story_length. each element is:[batch_size  ,embedding_size]
        print("3.1.candidate_list:",candidate_list)
        m_current=self.query_embedding

        h_current = tf.zeros((self.batch_size, self.hidden_size))
        #for each candidate sentence in the list,do loop.
        for pass_number in range(self.num_pass):
            # 1. attention mechansim.take fact representation c,question q,previous memory m_previous
            g = self.attention_mechanism_parallel(self.story_embedding, m_current,self.query_embedding)  # [batch_size,story_length]
            print("3.2.gate1 possibility distribution:", g) #[batch_size,story_length]


            if self.use_gated_gru:
                g = tf.split(g, self.story_length,axis=1)  # a list. length is: sequence_length. each element is:[batch_size,1]
                print("3.3.gate2 possibility distribution:", g)
                # 2. use gated-gru to update hidden state
                for i,c_current in enumerate(candidate_list):
                    g_current=g[i] #[batch_size,1]
                    h_current=self.gated_gru(c_current,h_current,g_current) #h_current:[batch_size,hidden_size]. g[i] represent score( a scalar) for current candidate sentence:c_current.

                #3. assign last hidden state to e(episodic)
                e_i=h_current #[batch_size,hidden_size]
                iii = 0
                iii / 0

            else: #for question answering
                #compuate e_i directly using weighted sum of candidate sentence. weight is given by gate.
                p_gate=tf.nn.softmax(g,dim=1)            #[batch_size,story_length]. compute weight
                p_gate=tf.expand_dims(p_gate,axis=2)     #[batch_size,story_length,1]
                e_i=tf.multiply(p_gate,self.story_embedding) #[batch_size,story_length,hidden_size]
                print("e_i:",e_i) #[batch_size,story_length,hidden_size]=(8, 3, 100)

            #4. use gru to update episodic memory m_i
            print("pass_number:",pass_number)
            print("====", e_i, m_current,"gru_episodic_memory")
            m_current=self.gru_cell(self, e_i, m_current,"gru_episodic_memory") #[batch_size,hidden_size]
            print("m_current:", m_current)
            iii = 0
            iii / 0
        self.m_T=m_current #[batch_size,hidden_size]
        print("3.episodic_memory_module.self.m_T:",self.m_T)

    def answer_module(self):
        """ Answer Module:generate an answer from the final memory vector.
        Input:
            hidden state from episodic memory module:[batch_size,hidden_size]
            question:[batch_size, embedding_size]
        """
        steps=self.sequence_length if self.sequence_decoder else 1 #decoder for a list of tokens with sequence. e.g."x1 x2 x3 x4..."
        a=self.m_T #init hidden state
        y_pred=tf.ones((self.batch_size,self.hidden_size)) #TODO usually we will init this as a special token '<GO>', you can change this line by pass embedding of '<GO>' from outside.
        logits_list=[]
        for i in range(steps):
            cell = rnn.GRUCell(self.hidden_size)
            y_previous_q=tf.concat([y_pred,self.query_embedding]) #[batch_hidden_size*2]
            _, a = cell( y_previous_q,a)
            logits=tf.matmul(tf.layers.dense(a,units=self.num_classes)) #[batch_size,vocab_size]

            #y_pred=tf.nn.softmax(logits) #[batch_size,vocab_size]
            logits_list.append(logits)

        if not self.sequence_decoder:
            self.logits=logits_list[0]
        else:
            self.logits=tf.stack(logits_list,axis=1) #[batch_size,sequence_length,vocab_size]

    def gated_gru(self,c_current,h_previous,g_current):
        """
        gated gru to get updated hidden state
        :param  c_current: [batch_size,embedding_size]
        :param  h_previous:[batch_size,hidden_size]
        :param  g_current: [batch_size,1]
        :return h_current: [batch_size,hidden_size]
        """
        # 1.compute candidate hidden state using GRU.
        h_candidate=self.gru_cell(c_current, h_previous,"gru_candidate_sentence") #[batch_size,hidden_size]
        # 2.combine candidate hidden state and previous hidden state using weight(a gate) to get updated hidden state.
        h_current=tf.multiply(g_current,h_candidate)+tf.multiply(1-g_current,h_previous) #[batch_size,hidden_size]
        return h_current

    def attention_mechanism_parallel(self,c_full,m,q):
        """ parallel implemtation of gate function given a list of candidate sentence, a query, and previous memory.
        Input:
           c_full: candidate fact. shape:[batch_size,story_length,hidden_size]
           m: previous memory. shape:[batch_size,hidden_size]
           q: question. shape:[batch_size,hidden_size]
        Output: a scalar score (in batch). shape:[batch_size,story_length]
        """
        q=tf.expand_dims(q,axis=1) #[batch_size,1,hidden_size]
        m=tf.expand_dims(m,axis=1) #[batch_size,1,hidden_size]

        # 1.define a large feature vector that captures a variety of similarities between input,memory and question vector: z(c,m,q)
        c_q_elementwise=tf.multiply(c_full,q)          #[batch_size,story_length,hidden_size]
        c_m_elementwise=tf.multiply(c_full,m)          #[batch_size,story_length,hidden_size]
        c_q_minus=tf.abs(tf.subtract(c_full,q))        #[batch_size,story_length,hidden_size]
        c_m_minus=tf.abs(tf.subtract(c_full,m))        #[batch_size,story_length,hidden_size]
        # c_transpose Wq
        c_w_q=self.x1Wx2_parallel(c_full,q,"c_w_q")   #[batch_size,story_length,hidden_size]
        c_w_m=self.x1Wx2_parallel(c_full,m,"c_w_m")   #[batch_size,story_length,hidden_size]
        # c_transposeWm
        q_tile=tf.tile(q,[1,self.story_length,1])     #[batch_size,story_length,hidden_size]
        m_tile=tf.tile(m,[1,self.story_length,1])     #[batch_size,story_length,hidden_size]
        z=tf.concat([c_full,m_tile,q_tile,c_q_elementwise,c_m_elementwise,c_q_minus,c_m_minus,c_w_q,c_w_m],2) #[batch_size,story_length,hidden_size*9]
        # 2. two layer feed foward
        g=tf.layers.dense(z,self.hidden_size*3,activation=tf.nn.tanh)  #[batch_size,story_length,hidden_size*3]
        g=tf.layers.dense(g,1,activation=tf.nn.sigmoid)                #[batch_size,story_length,1]
        g=tf.squeeze(g,axis=2)                                         #[batch_size,story_length]
        return g

    def x1Wx2_parallel(self,x1,x2,scope):
        """
        :param x1: [batch_size,story_length,hidden_size]
        :param x2: [batch_size,1,hidden_size]
        :param scope: a string
        :return:  [batch_size,story_length,hidden_size]
        """
        with tf.variable_scope(scope):
            x1=tf.reshape(x1,shape=(self.batch_size,-1)) #[batch_size,story_length*hidden_size]
            x1_w=tf.layers.dense(x1,self.story_length*self.hidden_size,use_bias=False) #[self.hidden_size, story_length*self.hidden_size]
            x1_w_expand=tf.expand_dims(x1_w,axis=2)     #[batch_size,story_length*self.hidden_size,1]
            x1_w_x2=tf.matmul(x1_w_expand,x2)           #[batch_size,story_length*self.hidden_size,hidden_size]
            x1_w_x2=tf.reshape(x1_w_x2,shape=(self.batch_size,self.story_length,self.hidden_size,self.hidden_size))
            x1_w_x2=tf.reduce_sum(x1_w_x2,axis=3)      #[batch_size,story_length,hidden_size]
            return x1_w_x2

    def gru_cell(self, Xt, h_t_minus_1,variable_scope):
        """
        single step of gru
        :param Xt: Xt:[batch_size,hidden_size]
        :param h_t_minus_1:[batch_size,hidden_size]
        :return:[batch_size,hidden_size]
        """
        with tf.variable_scope(variable_scope):
            # 1.update gate: decides how much past information is kept and how much new information is added.
            z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1,self.U_z) + self.b_z)  # z_t:[batch_size,self.hidden_size]
            # 2.reset gate: controls how much the past state contributes to the candidate state.
            r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1,self.U_r) + self.b_r)  # r_t:[batch_size,self.hidden_size]
            # 3.compute candiate state h_t~
            h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) +r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)  # h_t_candiate:[batch_size,self.hidden_size]
            # 4.compute new state: a linear combine of pervious hidden state and the current new state h_t~
            h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size,hidden_size]
        return h_t

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

    def loss_multilabel(self, l2_lambda=0.0001): #0.0001 this loss function is for multi-label classification
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
        with tf.variable_scope("gru_cell"):
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
    model = DynamicMemoryNetwork(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
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
    model = DynamicMemoryNetwork(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
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
            answer_single = np.sum(query, axis=1) + np.round(0.1 * np.sum(np.sum(story, axis=1),axis=1))  # [batch_size].e.g. np.array([1, 0, 1, 1, 1, 2, 1, 1])
            predict = sess.run([model.predictions], feed_dict={model.query: query, model.story: story,
                                                               model.dropout_keep_prob: dropout_keep_prob})
            print(i, "query:", query, "=====================>")
            print(i, "label:", answer_single, "prediction:", predict)

test()
#predict()