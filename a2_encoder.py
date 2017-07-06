# -*- coding: utf-8 -*-
"""
encoder for the transformer:
6 layers.each layers has two sub-layers.
the first is multi-head self-attention mechanism;
the second is position-wise fully connected feed-forward network.
for each sublayer. use LayerNorm(x+Sublayer(x)). all dimension=512.
"""
#TODO LAYER NORMALIZATION
import tensorflow as tf
from a2_base_model import BaseClass
import time
class Encoder(BaseClass):
    def __init__(self,d_model,d_k,d_v,sequence_length,h,batch_size,num_layer,Q,K_s,type='encoder',mask=None,dropout_keep_prob=None):
        """
        :param d_model:
        :param d_k:
        :param d_v:
        :param sequence_length:
        :param h:
        :param batch_size:
        :param embedded_words: shape:[batch_size*sequence_length,embed_size]
        """
        super(Encoder, self).__init__(d_model,d_k,d_v,sequence_length,h,batch_size,num_layer=num_layer)
        self.Q=Q
        self.K_s=K_s
        self.type=type
        self.mask=mask
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.dropout_keep_prob=dropout_keep_prob

    def encoder_fn(self):
        start = time.time()
        print("encoder_fn.started.")
        Q=self.Q
        K_s=self.K_s
        for layer_index in range(self.num_layer):
            Q, K_s=self.encoder_single_layer(Q,K_s,layer_index)
            print("encoder_fn.",layer_index,".Q:",Q,";K_s:",K_s)
        end = time.time()
        print("encoder_fn.ended.Q:",Q,";K_s:",K_s,";time spent:",(end-start))
        return Q,K_s

    def encoder_single_layer(self,Q,K_s,layer_index):
        """
        singel layer for encoder.each layers has two sub-layers:
        the first is multi-head self-attention mechanism; the second is position-wise fully connected feed-forward network.
        for each sublayer. use LayerNorm(x+Sublayer(x)). input and output of last dimension: d_model
        :param Q: shape should be:       [batch_size*sequence_length,d_model]
        :param K_s: shape should be:     [batch_size*sequence_length,d_model]
        :return:output: shape should be:[batch_size*sequence_length,d_model]
        """
        #1.1 the first is multi-head self-attention mechanism
        multi_head_attention_output=self.sub_layer_multi_head_attention(layer_index,Q,K_s,self.type,mask=self.mask,dropout_keep_prob=self.dropout_keep_prob) #[batch_size,sequence_length,d_model]
        #1.2 use LayerNorm(x+Sublayer(x)). all dimension=512.
        multi_head_attention_output=self.sub_layer_layer_norm_residual_connection(K_s ,multi_head_attention_output,layer_index,'encoder_multi_head_attention',dropout_keep_prob=self.dropout_keep_prob)

        #2.1 the second is position-wise fully connected feed-forward network.
        postion_wise_feed_forward_output=self.sub_layer_postion_wise_feed_forward(multi_head_attention_output,layer_index,self.type)
        #2.2 use LayerNorm(x+Sublayer(x)). all dimension=512.
        postion_wise_feed_forward_output= self.sub_layer_layer_norm_residual_connection(multi_head_attention_output,postion_wise_feed_forward_output,layer_index,'encoder_postion_wise_ff',dropout_keep_prob=self.dropout_keep_prob)
        return  postion_wise_feed_forward_output,postion_wise_feed_forward_output


def init():
    #1. assign value to fields
    vocab_size=1000
    d_model = 512
    d_k = 64
    d_v = 64
    sequence_length = 5*10
    h = 8
    batch_size=4*32
    initializer = tf.random_normal_initializer(stddev=0.1)
    # 2.set values for Q,K,V
    vocab_size=1000
    embed_size=d_model
    Embedding = tf.get_variable("Embedding_E", shape=[vocab_size, embed_size],initializer=initializer)
    input_x = tf.placeholder(tf.int32, [batch_size,sequence_length], name="input_x") #[4,10]
    print("input_x:",input_x)
    embedded_words = tf.nn.embedding_lookup(Embedding, input_x) #[batch_size*sequence_length,embed_size]
    Q = embedded_words  # [batch_size*sequence_length,embed_size]
    K_s = embedded_words  # [batch_size*sequence_length,embed_size]
    num_layer=6
    mask = get_mask(batch_size, sequence_length)
    #3. get class object
    encoder_class=Encoder(d_model,d_k,d_v,sequence_length,h,batch_size,num_layer,Q,K_s,mask=mask) #Q,K_s,embedded_words
    return encoder_class,Q,K_s

def get_mask(batch_size,sequence_length):
    lower_triangle=tf.matrix_band_part(tf.ones([sequence_length,sequence_length]),-1,0)
    result=-1e9*(1.0-lower_triangle)
    print("get_mask==>result:",result)
    return result

def test_sub_layer_multi_head_attention(encoder_class,index_layer,Q,K_s):
    sub_layer_multi_head_attention_output=encoder_class.sub_layer_multi_head_attention(index_layer,Q,K_s)
    return sub_layer_multi_head_attention_output

def test_postion_wise_feed_forward(encoder_class,x,layer_index):
    sub_layer_postion_wise_feed_forward_output=encoder_class.sub_layer_postion_wise_feed_forward(x, layer_index)
    return sub_layer_postion_wise_feed_forward_output

encoder_class,Q,K_s=init()
#index_layer=0

#below is 4 callable codes for testing functions:from small function to whole function of encoder.

#1.test 1: for sub layer of multi head attention
#sub_layer_multi_head_attention_output=test_sub_layer_multi_head_attention(encoder_class,index_layer,Q,K_s)
#print("sub_layer_multi_head_attention_output1:",sub_layer_multi_head_attention_output)

#2. test 2: for sub layer of multi head attention with poistion-wise feed forward
#d1,d2,d3=sub_layer_multi_head_attention_output.get_shape().as_list()
#postion_wise_ff_input=tf.reshape(sub_layer_multi_head_attention_output,shape=[-1,d3])
#sub_layer_postion_wise_feed_forward_output=test_postion_wise_feed_forward(encoder_class,postion_wise_ff_input,index_layer)
#sub_layer_postion_wise_feed_forward_output=tf.reshape(sub_layer_postion_wise_feed_forward_output,shape=(d1,d2,d3))
#print("sub_layer_postion_wise_feed_forward_output2:",sub_layer_postion_wise_feed_forward_output)

#3.test 3: test for single layer of encoder
#encoder_class.encoder_single_layer(Q,K_s,index_layer)

#4.test 4: test for encoder. with N layers

#Q,K_s = encoder_class.encoder_fn()