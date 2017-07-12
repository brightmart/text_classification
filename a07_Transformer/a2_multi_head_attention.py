# -*- coding: utf-8 -*-
#test self-attention
import tensorflow as tf
import time
"""
multi head attention.
1.linearly project the queries,keys and values h times(with different,learned linear projections to d_k,d_k,d_v dimensions)
2.scaled dot product attention for each projected version of Q,K,V
3.concatenated result
4.linear projection to get final result

three kinds of usage:
1. attention for encoder
2. attention for decoder(need a mask to pay attention for only known position)
3. attention as bridge of encoder and decoder
"""
class MultiHeadAttention(object):
    """ multi head attention"""
    def __init__(self,Q,K_s,V_s,d_model,d_k,d_v,sequence_length,h,type=None,is_training=None,mask=None,dropout_rate=0.1):
        self.d_model=d_model
        self.d_k=d_k
        self.d_v=d_v
        self.sequence_length=sequence_length
        self.h=h
        self.Q=Q
        self.K_s=K_s
        self.V_s=V_s
        self.type=type
        self.is_training=is_training
        self.mask=mask
        self.dropout_rate=dropout_rate
        print("MultiHeadAttention.self.dropout_rate:",self.dropout_rate)

    def multi_head_attention_fn(self):
        """
        multi head attention
        :param Q: query.  shape:[batch,sequence_length,d_model]
        :param K_s: keys. shape:[batch,sequence_length,d_model].
        :param V_s:values.shape:[batch,sequence_length,d_model].
        :param h: h times
        :return: result of scaled dot product attention. shape:[sequence_length,d_model]
        """
        # 1. linearly project the queries,keys and values h times(with different,learned linear projections to d_k,d_k,d_v dimensions)
        Q_projected   = tf.layers.dense(self.Q,units=self.d_model)     # [batch,sequence_length,d_model]
        K_s_projected = tf.layers.dense(self.K_s, units=self.d_model)  # [batch,sequence_length,d_model]
        V_s_projected = tf.layers.dense(self.V_s, units=self.d_model)  # [batch,sequence_length,d_model]
        # 2. scaled dot product attention for each projected version of Q,K,V
        dot_product=self.scaled_dot_product_attention_batch(Q_projected,K_s_projected,V_s_projected) # [batch,h,sequence_length,d_k]
        # 3. concatenated
        print("dot_product:====================================================================================>",dot_product) #dot_product:(128, 8, 6, 64)
        batch_size,h,length,d_k=dot_product.get_shape().as_list()
        print("self.sequence_length:",self.sequence_length) #5
        dot_product=tf.reshape(dot_product,shape=(-1,length,self.d_model))
        # 4. linear projection
        output=tf.layers.dense(dot_product,units=self.d_model) # [batch,sequence_length,d_model]
        return output  #[batch,sequence_length,d_model]

    def scaled_dot_product_attention_batch_mine(self,Q,K_s,V_s): #my own implementation of scaled dot product attention.
        """
        scaled dot product attention
        :param Q:  query.  shape:[batch,sequence_length,d_model]
        :param K_s: keys.  shape:[batch,sequence_length,d_model]
        :param V_s:values. shape:[batch,sequence_length,d_model]
        :param mask:       shape:[batch,sequence_length]
        :return: result of scaled dot product attention. shape:[batch,h,sequence_length,d_k]
        """
        # 1. split Q,K,V
        Q_heads = tf.stack(tf.split(Q,self.h,axis=2),axis=1)         # [batch,h,sequence_length,d_k]
        K_heads = tf.stack(tf.split(K_s, self.h, axis=2), axis=1)    # [batch,h,sequence_length,d_k]
        V_heads = tf.stack(tf.split(V_s, self.h, axis=2), axis=1)    # [batch,h,sequence_length,d_k]
        dot_product=tf.multiply(Q_heads,K_heads)                     # [batch,h,sequence_length,d_k]
        # 2. dot product
        dot_product=dot_product*(1.0/tf.sqrt(tf.cast(self.d_model,tf.float32))) # [batch,h,sequence_length,d_k]
        dot_product=tf.reduce_sum(dot_product,axis=-1,keep_dims=True) # [batch,h,sequence_length,1]
        # 3. add mask if it is none
        if self.mask is not None:
            mask = tf.expand_dims(self.mask, axis=-1)  # [batch,sequence_length,1]
            mask = tf.expand_dims(mask, axis=1)  # [batch,1,sequence_length,1]
            dot_product=dot_product+mask   # [batch,h,sequence_length,1]
        # 4. get possibility
        p=tf.nn.softmax(dot_product)                                  # [batch,h,sequence_length,1]
        # 5. final output
        output=tf.multiply(p,V_heads)                                 # [batch,h,sequence_length,d_k]
        return output                                                 # [batch,h,sequence_length,d_k]

    def scaled_dot_product_attention_batch(self, Q, K_s, V_s):# scaled dot product attention: implementation style like tensor2tensor from google
        """
        scaled dot product attention
        :param Q:  query.  shape:[batch,sequence_length,d_model]
        :param K_s: keys.  shape:[batch,sequence_length,d_model]
        :param V_s:values. shape:[batch,sequence_length,d_model]
        :param mask:       shape:[sequence_length,sequence_length]
        :return: result of scaled dot product attention. shape:[batch,h,sequence_length,d_k]
        """
        # 1. split Q,K,V
        Q_heads = tf.stack(tf.split(Q,self.h,axis=2),axis=1)                    # [batch,h,sequence_length,d_k]
        K_heads = tf.stack(tf.split(K_s, self.h, axis=2), axis=1)               # [batch,h,sequence_length,d_k]
        V_heads = tf.stack(tf.split(V_s, self.h, axis=2), axis=1)               # [batch,h,sequence_length,d_k]
        # 2. dot product of Q,K
        dot_product=tf.matmul(Q_heads,K_heads,transpose_b=True)                 # [batch,h,sequence_length,sequence_length]
        dot_product=dot_product*(1.0/tf.sqrt(tf.cast(self.d_model,tf.float32))) # [batch,h,sequence_length,sequence_length]
        # 3. add mask if it is none
        print("scaled_dot_product_attention_batch.===============================================================>mask is not none?",self.mask is not None)
        if self.mask is not None:
            mask_expand=tf.expand_dims(tf.expand_dims(self.mask,axis=0),axis=0) # [1,1,sequence_length,sequence_length]
            #dot_product:(128, 8, 6, 6);mask_expand:(1, 1, 5, 5)
            print("scaled_dot_product_attention_batch.===============================================================>dot_product:",dot_product,";mask_expand:",mask_expand)
            dot_product=dot_product+mask_expand                                 # [batch,h,sequence_length,sequence_length]
        # 4.get possibility
        weights=tf.nn.softmax(dot_product)                                      # [batch,h,sequence_length,sequence_length]
        # drop out weights
        weights=tf.nn.dropout(weights,1.0-self.dropout_rate)                    # [batch,h,sequence_length,sequence_length]
        # 5. final output
        output=tf.matmul(weights,V_heads)                                       # [batch,h,sequence_length,d_model]
        return output


#vectorized implementation of multi head attention for sentences with batch
def multi_head_attention_for_sentence_vectorized(layer_number):
    print("started...")
    start = time.time()
    # 1.set parameter
    d_model = 512
    d_k = 64
    d_v = 64
    sequence_length = 1000
    h = 8
    batch_size=128
    initializer = tf.random_normal_initializer(stddev=0.1)
    # 2.set Q,K,V
    vocab_size=1000
    embed_size=d_model
    type='decoder'
    Embedding = tf.get_variable("Embedding_", shape=[vocab_size, embed_size],initializer=initializer)
    input_x = tf.placeholder(tf.int32, [batch_size,sequence_length], name="input_x")
    embedded_words = tf.nn.embedding_lookup(Embedding, input_x) #[batch_size,sequence_length,embed_size]
    mask=get_mask(batch_size,sequence_length) #tf.ones((batch_size,sequence_length))*-1e8  #[batch,sequence_length]
    with tf.variable_scope("query_at_each_sentence"+str(layer_number)):
        Q = embedded_words  # [batch_size*sequence_length,embed_size]
        K_s=embedded_words #[batch_size*sequence_length,embed_size]
        V_s=tf.get_variable("V_s_original_", shape=embedded_words.get_shape().as_list(),initializer=initializer) #[batch_size,sequence_length,embed_size]
        # 3.call method to get result
        multi_head_attention_class = MultiHeadAttention(Q, K_s, V_s, d_model, d_k, d_v, sequence_length, h,type='decoder',mask=mask)
        encoder_output=multi_head_attention_class.multi_head_attention_fn() #shape:[sequence_length,d_model]
        encoder_output=tf.reshape(encoder_output,shape=(batch_size,sequence_length,d_model))
    end = time.time()
    print("input_x:",input_x)
    print("encoder_output:",encoder_output,";time_spent:",(end-start))

def get_mask(batch_size,sequence_length):
    lower_triangle=tf.matrix_band_part(tf.ones([sequence_length,sequence_length]),-1,0)
    result=-1e9*(1.0-lower_triangle)
    print("get_mask==>result:",result)
    return result

#multi_head_attention_for_sentence_vectorized(0)