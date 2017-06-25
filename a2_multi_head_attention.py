#test self-attention
import tensorflow as tf
"""
multi head attention.
1.linearly project the queries,keys and values h times(with different,learned linear projections to d_k,d_k,d_v dimensions)
2.scaled dot product attention for each projected version of Q,K,V
3.concatenated result
4.linear projection to get final result
"""
class MultiHeadAttention(object):
    """ multi head attention"""
    def __init__(self,Q,K_s,V_s,d_model,d_k,d_v,sequence_length,h,mask=None):
        print("multi_head_attention.init.started")
        self.d_model=d_model
        self.d_k=d_k
        self.d_v=d_v
        self.sequence_length=sequence_length
        self.h=h
        self.Q=Q
        self.K_s=K_s
        self.V_s=V_s
        self.mask=mask #mask is a list. length is sequence_length. each element is a scaler value. e.g. [1,1,1,-1000000,-1000000,-1000000,....-1000000]
        print("multi_head_attention.init.ended")

    def multi_head_attention_fn(self):
        """
        multi head attention
        :param Q: query. [sequence_length,d_k]
        :param K_s: keys. shape:[sequence_length,d_k].
        :param V_s:values.shape:[sequence_length,d_v].
        :param h: h times
        :return: result of scaled dot product attention. shape:[sequence_length,d_k]
        """
        print("multi_head_attention.multi_head_attention_fn.started")
        # 1. linearly project the queries,keys and values h times(with different,learned linear projections to d_k,d_k,d_v dimensions)
        Q_list, K_s_list, V_s_list = self.linear_projection()
        weight_sum_attention_list=[]
        # 2.scaled dot product attention for each projected version of Q,K,V
        for i in range(self.h):
            Q = Q_list[i]
            K_s = K_s_list[i]
            V_s = V_s_list[i]
            scaled_dot_product_attention_=self.scaled_dot_product_attention(Q,K_s,V_s) #[sequence_length,d_v]
            weight_sum_attention_list.append(scaled_dot_product_attention_)
        # 3. concatenated
        weight_sum_attention=tf.concat(weight_sum_attention_list,axis=1) #[sequence_length,d_v*h]
        print("weight_sum_attention:====>",weight_sum_attention)
        # 4. linear projection
        with tf.variable_scope("multi_head_attention_projection"):
            W_multi_head_attention_projection = tf.get_variable("W_multi_head_attention_projection",shape=[self.d_v * self.h, self.d_model])
            b_multi_head_attention_projection = tf.get_variable("b_multi_head_attention_projection",shape=[self.d_model])
        weight_sum_attention_projected = tf.matmul(weight_sum_attention,W_multi_head_attention_projection) + b_multi_head_attention_projection  # [sequence_length,d_model]
        print("multi_head_attention.multi_head_attention_fn.ended")
        return weight_sum_attention_projected  # [sequence_length,d_model]

    def scaled_dot_product_attention(self,Q,K_s,V_s):
        """
        scaled dot product attention
        :param Q: query.[sequence_length,d_k]
        :param K_s: keys. shape:[sequence_length,d_k].
        :param V_s:values.shape:[sequence_length,d_v].
        :return: result of scaled dot product attention. shape:[sequence_length,d_v]
        """
        print("scaled_dot_product_attention.@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Q:",Q,";K_s:",K_s,";V_s:",V_s)
        q_shape_list=Q.get_shape().as_list()
        k_shape_list=K_s.get_shape().as_list()
        if q_shape_list!=k_shape_list: # special design for attention between encoder and decoder, where sequence length is different between each other.
            Q_K_dot=self.Q_K_different_shape(Q,K_s)
        else:
            d_k=q_shape_list[0] if len(q_shape_list)==1 else q_shape_list[1]
            Q_K_dot=tf.multiply(Q, K_s) #[sequence_length,d_model].
            Q_K_dot=tf.reduce_sum(Q_K_dot,axis=1) #[sequence_length].
            Q_K_dot=tf.div(Q_K_dot,tf.sqrt(tf.cast(d_k,dtype=tf.float32))) #[sequence_length].
            if self.mask is not None: #mask is a list.
                Q_K_dot=Q_K_dot*self.mask
        p_attention=tf.nn.softmax(Q_K_dot) #[sequence_length]
        p_attention=tf.expand_dims(p_attention,axis=1) #[sequence_length,1]
        weight_sum_attention=tf.multiply(p_attention,V_s) #[sequence_length,d_model]
        return weight_sum_attention #[sequence_length,d_v]

    def Q_K_different_shape(self,Q,K):
        Q_expand = tf.expand_dims(Q, axis=1) #[sequence_length1,1,d_q]
        K_expand = tf.expand_dims(K, axis=0) #[1,sequence_length2,d_q]
        Q_K_dot=tf.multiply(Q_expand,K_expand) #[sequence_length1,sequence_length2,d_q]
        Q_K_dot=tf.reduce_sum(Q_K_dot,axis=1) #[sequence_length1,d_q]
        Q_K_dot=tf.reduce_sum(Q_K_dot,axis=1) #[sequence_length1]=[sequnce_length_q]
        return Q_K_dot

    def linear_projection(self):
        """
        linear projection for Q,K,V for h times to get h differenet representation of Q,K,V
        :param Q: query. [sequence_length,d_model]
        :param K_s: keys. shape:[sequence_length,d_k].
        :param V_s:values.shape:[sequence_length,d_v].
        :return: result of scaled dot product attention. shape:[sequence_length,d_k]
        """
        Q_list=[]
        K_s_list=[]
        V_s_list=[]
        for i in range(self.h):
            Q_, K_, V_=self.linear_projection_single(i)
            Q_list.append(Q_)
            K_s_list.append(K_)
            V_s_list.append(V_)
        return Q_list,K_s_list,V_s_list

    def linear_projection_single(self,i):
        """
        get a single linear projection for Q,K,V
        :param Q: query. [sequence_length,d_model]
        :param K_s: keys. shape:[sequence_length,d_k].
        :param V_s:values.shape:[sequence_length,d_v].
        :param i: scalar. index of projection.(totally h times projections)
        :return:Q_projected:[sequence_length,self.d_model];
                K_projected:[sequence_length,self.d_model];
                V_projected:[sequence_length,self.d_model]
        """
        with tf.variable_scope("linear_projection_single"):
            W_q_i=tf.get_variable("W_q_i"+str(i),shape=[self.d_model,self.d_k])
            W_k_i=tf.get_variable("W_k_i"+str(i),shape=[self.d_model,self.d_k])
            W_v_i=tf.get_variable("W_v_i"+str(i),shape=[self.d_model,self.d_v])
        Q_projected=tf.matmul(self.Q,W_q_i)   #[sequence_length,self.d_model]<------------matmul([sequence_length,self.d_k],[self.d_k,self.d_model])
        K_projected=tf.matmul(self.K_s,W_k_i) #[sequence_length,self.d_model]<------------matmul([sequence_length,self.d_k],[self.d_k,self.d_model])
        V_projected=tf.matmul(self.V_s,W_v_i) #[sequence_length,self.d_model]<------------matmul([sequence_length,self.d_v],[self.d_v,self.d_model)

        return Q_projected,K_projected,V_projected

#vanilla implement of multi head attention for sentences with batch
def multi_head_attention_for_sentence_vanilla():
    print("started...")
    # 1.set parameter
    d_model = 512
    d_k = 64
    d_v = 64
    sequence_length = 10
    h =8
    batch_size=4
    initializer = tf.random_normal_initializer(stddev=0.1)
    # 2.set Q,K,V
    vocab_size=1000
    embed_size=100
    Embedding = tf.get_variable("Embedding_A", shape=[vocab_size, embed_size],initializer=initializer)
    input_x = tf.placeholder(tf.int32, [batch_size,sequence_length], name="input_x") #[4,10]
    input_x_=tf.reshape(input_x,(batch_size*sequence_length,)) #[batch_size*sequence_length]
    embedded_words = tf.nn.embedding_lookup(Embedding, input_x_) #[batch_size*sequence_length,embed_size]
    K_s=embedded_words #TODO
    V_s=tf.get_variable("V_s_original", shape=embedded_words.get_shape().as_list(),initializer=initializer) #embedded_words #TODO
    embedded_words_list=tf.split(embedded_words,batch_size*sequence_length,axis=0) #list.length is sequence_length.each element is:[ 1,embed_size]
    embedded_words_list=[tf.squeeze(x,axis=0) for x in embedded_words_list]          #list.length is sequence_length.each element is:[   embed_size]

    with tf.variable_scope("query_at_each_sentence"):
        encoder_list=[]
        for i,Q in enumerate(embedded_words_list):#q's shape:[embed_size]
            if i>0:
                tf.get_variable_scope().reuse_variables()
            # 3.call method to get result
            multi_head_attention_class = MultiHeadAttention(Q, K_s, V_s, d_model, d_k, d_v, sequence_length, h)
            multi_head_attention_result=multi_head_attention_class.multi_head_attention_fn() #shape:[sequence_length,d_model]
            print(i,"result:",multi_head_attention_result) #shape:[sequence_length,d_model]
            multi_head_attention_result=tf.reduce_sum(multi_head_attention_result,axis=0) #shape:[d_model]
            encoder_list.append(multi_head_attention_result)
    encoder_output=tf.stack(encoder_list,axis=0) #[sequence_length,d_model]
    encoder_output=tf.reshape(encoder_output,shape=(batch_size,sequence_length,d_model))
    print("input_x:",input_x) #(4, 10),
    print("encoder_output:",encoder_output) #(4, 10, 512)

#vectorized implementation of multi head attention for sentences with batch
def multi_head_attention_for_sentence_vectorized(layer_number):
    print("started...")
    # 1.set parameter
    d_model = 512
    d_k = 64
    d_v = 64
    sequence_length = 10
    h = 8
    batch_size=4
    initializer = tf.random_normal_initializer(stddev=0.1)
    # 2.set Q,K,V
    vocab_size=1000
    embed_size=d_model
    Embedding = tf.get_variable("Embedding_", shape=[vocab_size, embed_size],initializer=initializer)
    input_x = tf.placeholder(tf.int32, [batch_size,sequence_length], name="input_x")
    input_x_=tf.reshape(input_x,(batch_size*sequence_length,)) #[batch_size*sequence_length]
    embedded_words = tf.nn.embedding_lookup(Embedding, input_x_) #[batch_size*sequence_length,embed_size]
    with tf.variable_scope("query_at_each_sentence"+str(layer_number)):
        Q = embedded_words  # [batch_size*sequence_length,embed_size]
        K_s=embedded_words #[batch_size*sequence_length,embed_size]
        V_s=tf.get_variable("V_s_original_", shape=embedded_words.get_shape().as_list(),initializer=initializer) #[batch_size*sequence_length,embed_size]
        # 3.call method to get result
        multi_head_attention_class = MultiHeadAttention(Q, K_s, V_s, d_model, d_k, d_v, sequence_length, h)
        encoder_output=multi_head_attention_class.multi_head_attention_fn() #shape:[sequence_length,d_model]
        encoder_output=tf.reshape(encoder_output,shape=(batch_size,sequence_length,d_model))
    print("input_x:",input_x) #(4, 10),
    print("encoder_output:",encoder_output) #(4, 10, 512)

#multi_head_attention_for_sentence_vanilla()
#multi_head_attention_for_sentence_vectorized(0)