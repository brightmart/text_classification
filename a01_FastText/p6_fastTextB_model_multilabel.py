# autor:xul
# fast text. using: very simple model;n-gram to captrue location information;h-softmax to speed up training/inference
# for the n-gram you can use data_util to generate. see method process_one_sentence_to_get_ui_bi_tri_gram under aa1_data_util/data_util_zhihu.py
import tensorflow as tf

class fastTextB:
    def __init__(self, label_size, learning_rate, batch_size, decay_steps, decay_rate,num_sampled,sentence_len,vocab_size,embed_size,is_training,max_label_per_example=5):
        """init all hyperparameter here"""
        # 1.set hyper-paramter
        self.label_size = label_size #e.g.1999
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.sentence_len=sentence_len
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.is_training=is_training
        self.learning_rate=learning_rate
        self.max_label_per_example=max_label_per_example
        self.initializer=tf.random_normal_initializer(stddev=0.1)

        # 2.add placeholder (X,label)
        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")     #X
        self.labels = tf.placeholder(tf.int64, [None,self.max_label_per_example], name="Labels") #y [1,2,3,3,3]
        self.labels_l1999=tf.placeholder(tf.float32,[None,self.label_size]) # int64
        #3.set some variables
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0, trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")

        #4.init weights
        self.instantiate_weights()
        #5.main graph: inference
        self.logits = self.inference() #[None, self.label_size]
        #6.calculate loss
        self.loss_val = self.loss()
        #7.start training by update parameters using according loss
        self.train_op = self.train()

        #8.calcuate accuracy
        # correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.labels) #2.TODO tf.argmax(self.logits, 1)-->[batch_size]
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") #TODO

    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size],initializer=self.initializer)
        self.W = tf.get_variable("W", [self.embed_size, self.label_size],initializer=self.initializer)
        self.b = tf.get_variable("b", [self.label_size])

    def inference(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1.get emebedding of words in the sentence
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding,self.sentence)  # [None,self.sentence_len,self.embed_size]

        # 2.average vectors, to get representation of the sentence
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)  # [None,self.embed_size]

        # 3.linear classifier layer
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b #[None, self.label_size]==tf.matmul([None,self.embed_size],[self.embed_size,self.label_size])
        return logits


    def loss(self,l2_lambda=0.0001):
        """calculate loss using (NCE)cross entropy here"""
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        #if self.is_training:#training
            #labels=tf.reshape(self.labels,[-1])               #3.[batch_size,max_label_per_example]------>[batch_size*max_label_per_example,]
            #labels=tf.expand_dims(labels,1)                   #[batch_size*max_label_per_example,]----->[batch_size*max_label_per_example,1]
            #nce_loss: notice-->for now, if you have a variable number of target classes, you can pad them out to a constant number by either repeating them or by padding with an otherwise unused class.
         #   loss = tf.reduce_mean(#inputs's SHAPE should be: [batch_size, dim]
         #       tf.nn.nce_loss(weights=tf.transpose(self.W),  #[embed_size, label_size]--->[label_size,embed_size]. nce_weights:A `Tensor` of shape `[num_classes, dim].O.K.
         #                      biases=self.b,                 #[label_size]. nce_biases:A `Tensor` of shape `[num_classes]`.
         #                      labels=self.labels,                 #4.[batch_size,max_label_per_example]. train_labels, # A `Tensor` of type `int64` and shape `[batch_size,num_true]`. The target classes.
         #                      inputs=self.sentence_embeddings,#TODO [None,self.embed_size] #A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
         #                      num_sampled=self.num_sampled,  #  scalar. 100
         #                      num_true=self.max_label_per_example,
         #                      num_classes=self.label_size,partition_strategy="div"))  #scalar. 1999
        #else:#eval(/inference)
        labels_multi_hot = self.labels_l1999 #[batch_size,label_size]
        #sigmoid_cross_entropy_with_logits:Computes sigmoid cross entropy given `logits`.Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive.  For instance, one could perform multilabel classification where a picture can contain both an elephant and a dog at the same time.
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_multi_hot,logits=self.logits) #labels:[batch_size,label_size];logits:[batch, label_size]
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1)) # reduce_sum
        print("loss:",loss)

        # add regularization result in not converge
        self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        print("l2_losses:",self.l2_losses)
        loss=loss+self.l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op
