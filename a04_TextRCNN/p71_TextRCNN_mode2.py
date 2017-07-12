# -*- coding: utf-8 -*-
#TextRNN: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat output, 4.FC layer, 5.softmax
import tensorflow as tf
import numpy as np
import copy
class TextRCNN:
    def __init__(self,num_classes, learning_rate, decay_steps, decay_rate,sequence_length,
                 vocab_size,embed_size,is_training,batch_size,initializer=tf.random_normal_initializer(stddev=0.1),multi_label_flag=False):
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
        self.activation=tf.nn.relu #TODO tf.nn.tanh
        self.multi_label_flag=multi_label_flag

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32, [None,], name="input_y")  # y:[None,num_classes]
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        #self.batch_size=tf.placeholder(tf.int32,name="batch_size")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        if not self.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()
        else:
            self.accuracy = tf.constant(0.5) #fake accuracy. (you can calcuate accuracy outside of graph using method calculate_accuracy(...) in train.py)

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("weights"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)

            self.left_side_first_word= tf.get_variable("left_side_first_word",shape=[self.batch_size, self.embed_size],initializer=self.initializer) #TODO removed. replaced with zero vector
            self.right_side_last_word = tf.get_variable("right_side_last_word",shape=[self.batch_size, self.embed_size],initializer=self.initializer) #TODO removed. replaced with zero vector
            #self.left_side_context_first= tf.get_variable("left_side_context_first",shape=[self.batch_size, self.embed_size],initializer=self.initializer) #TODO removed. replaced with zero vector
            #self.right_side_context_last=tf.get_variable("right_side_context_last",shape=[self.batch_size, self.embed_size],initializer=self.initializer) #TODO removed. replaced with zero vector

            self.W_l=tf.get_variable("W_l",shape=[self.embed_size, self.embed_size],initializer=self.initializer)
            self.W_r=tf.get_variable("W_r",shape=[self.embed_size, self.embed_size],initializer=self.initializer)
            self.W_sl=tf.get_variable("W_sl",shape=[self.embed_size, self.embed_size],initializer=self.initializer)
            self.W_sr=tf.get_variable("W_sr",shape=[self.embed_size, self.embed_size],initializer=self.initializer)

            self.b = tf.get_variable("b", [self.embed_size])

            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*3, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size]

        #b = tf.get_variable("b", [self.embed_size*3])
        #h = tf.nn.relu(tf.nn.bias_add(output_conv, b), "relu")
    def get_context_left(self,context_left,embedding_previous):
        """
        :param context_left:
        :param embedding_previous:
        :return: output:[None,embed_size]
        """
        left_c=tf.matmul(context_left,self.W_l)       #shape:[batch_size,embed_size]<---------context_left:[batch_size,embed_size];W_l:[embed_size,embed_size]
        left_e=tf.matmul(embedding_previous,self.W_sl)#shape:[batch_size,embed_size]<---------embedding_previous;[batch_size,embed_size];W_sl:[embed_size, embed_size]
        left_h=left_c+left_e #shape:[batch_size,embed_size]
        #context_left=self.activation(left_h) #shape:[batch_size,embed_size] #TODO
        context_left = tf.nn.relu(tf.nn.bias_add(left_h, self.b), "relu") #TODO
        return context_left #shape:[batch_size,embed_size]

    def get_context_right(self,context_right,embedding_afterward):
        """
        :param context_right:
        :param embedding_afterward:
        :return: output:[None,embed_size]
        """
        right_c=tf.matmul(context_right,self.W_r)        #shape:[batch_size,embed_size]<---------context_right:[batch_size,embed_size];W_r:[embed_size,embed_size]
        right_e=tf.matmul(embedding_afterward,self.W_sr) #shape:[batch_size,embed_size]<----------embedding_afterward:[batch_size,embed_size];W_sr:[embed_size,embed_size]
        right_h=right_c+right_e #shape:[batch_size,embed_size]
        #context_right=self.activation(right_h) #shape:[batch_size,embed_size] #TODO
        context_right = tf.nn.relu(tf.nn.bias_add(right_h, self.b), "relu") #TODO
        return context_right #shape:[batch_size,embed_size]

    def conv_layer_with_recurrent_structure(self):
        """
        input:self.embedded_words:[None,sentence_length,embed_size]
        :return: shape:[None,sentence_length,embed_size*3]
        """
        #1. get splitted list of word embeddings
        embedded_words_split=tf.split(self.embedded_words,self.sequence_length,axis=1) #sentence_length个[None,1,embed_size]
        embedded_words_squeezed=[tf.squeeze(x,axis=1) for x in embedded_words_split]#sentence_length个[None,embed_size]
        embedding_previous=self.left_side_first_word #tf.zeros((self.batch_size,self.embed_size)) #TODO SHOULD WE ASSIGN A VARIABLE HERE
        context_left_previous=tf.zeros((self.batch_size,self.embed_size)) #self.left_side_context_first# TODO SHOULD WE ASSIGN A VARIABLE HERE

        #2. get list of context left
        context_left_list=[]
        for i,current_embedding_word in enumerate(embedded_words_squeezed):#sentence_length个[None,embed_size]
            context_left=self.get_context_left(context_left_previous, embedding_previous) #[None,embed_size]
            context_left_list.append(context_left) #append result to list
            embedding_previous=current_embedding_word #assign embedding_previous
            context_left_previous=context_left #assign context_left_previous

        #3. get context right
        embedded_words_squeezed2=copy.copy(embedded_words_squeezed)
        embedded_words_squeezed2.reverse()
        embedding_afterward=self.right_side_last_word #tf.zeros((self.batch_size,self.embed_size)) # TODO self.right_side_last_word SHOULD WE ASSIGN A VARIABLE HERE
        context_right_afterward = tf.zeros((self.batch_size, self.embed_size)) #self.right_side_context_last # TODO SHOULD WE ASSIGN A VARIABLE HERE
        context_right_list=[]
        for j,current_embedding_word in enumerate(embedded_words_squeezed2):
            context_right=self.get_context_right(context_right_afterward,embedding_afterward)
            context_right_list.append(context_right)
            embedding_afterward=current_embedding_word
            context_right_afterward=context_right

        #4.ensemble "left,embedding,right" to output
        output_list=[]
        for index,current_embedding_word in enumerate(embedded_words_squeezed):
            representation=tf.concat([context_left_list[index],current_embedding_word,context_right_list[index]],axis=1) #representation's shape:[None,embed_size*3]
            output_list.append(representation) #shape:sentence_length个[None,embed_size*3]

        #5. stack list to a tensor
        output=tf.stack(output_list,axis=1) #shape:[None,sentence_length,embed_size*3]
        return output


    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.max pooling, 4.FC layer 5.softmax """
        #1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x) #shape:[None,sentence_length,embed_size]
        #2. Bi-lstm layer
        output_conv=self.conv_layer_with_recurrent_structure() #shape:[None,sentence_length,embed_size*3]
        #2.1 apply nolinearity
        #b = tf.get_variable("b", [self.embed_size*3])
        #h = tf.nn.relu(tf.nn.bias_add(output_conv, b), "relu")

        #3. max pooling
        output_pooling=tf.reduce_max(output_conv,axis=1) #shape:[None,embed_size*3]
        #4. logits(use linear layer)
        with tf.name_scope("dropout"):
            h_drop=tf.nn.dropout(output_pooling,keep_prob=self.dropout_keep_prob) #[None,embed_size*3]

        with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            logits = tf.matmul(h_drop, self.W_projection) + self.b_projection  #shape:[batch_size,num_classes]<-----h_drop:[None,embed_size*3];b_projection:[hidden_size*3, self.num_classes]
        return logits

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

    def loss_multilabel(self,l2_lambda=0.00001): #0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            #input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits);#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            #losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, 1999).
            losses=tf.reduce_sum(losses,axis=1) #shape=(?,). loss for all data in the batch
            loss=tf.reduce_mean(losses)         #shape=().   average loss in the batch
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op

#test started
def test():
    #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes=10
    learning_rate=0.01
    batch_size=8
    decay_steps=1000
    decay_rate=0.9
    sequence_length=5
    vocab_size=10000
    embed_size=100
    is_training=True
    dropout_keep_prob=1#0.5
    textRNN=TextRCNN(num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
            input_y=input_y=np.array([1,0,1,1,1,2,1,1]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
            loss,acc,predict,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],
                                        feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
            print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
#test()