# -*- coding: utf-8 -*-
"""
Decoder:
1. The decoder is composed of a stack of N= 6 identical layers.
2. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
attention over the output of the encoder stack.
3. Similar to the encoder, we employ residual connections
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions.  This
masking, combined with fact that the output embeddings are offset by one position, ensures that the
predictions for position i can depend only on the known outputs at positions less than i.
"""
import tensorflow as tf
from a2_base_model import BaseClass
from a2_attention_between_enc_dec import AttentionEncoderDecoder
import time

class Decoder(BaseClass):
    def __init__(self,d_model,d_k,d_v,sequence_length,h,batch_size,Q,K_s,K_v_encoder,decoder_sent_length,
                 num_layer=6,type='decoder',is_training=True,mask=None,dropout_keep_prob=None):
        """
        :param d_model:
        :param d_k:
        :param d_v:
        :param sequence_length:
        :param h:
        :param batch_size:
        :param Q:
        :param K_s:
        :param K_v_encoder: shape:[batch_size,sequence_length,embed_size]. it is the output from encoder
        """
        super(Decoder, self).__init__(d_model, d_k, d_v, sequence_length, h, batch_size, num_layer=num_layer)
        self.Q=Q
        self.K_s=K_s
        self.K_v_encoder=K_v_encoder
        self.type=type
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.is_training=is_training
        self.decoder_sent_length=decoder_sent_length
        self.mask=mask
        self.dropout_keep_prob=dropout_keep_prob

    def decoder_fn(self):
        start = time.time()
        print("decoder.decoder_fn.started.")
        Q=self.Q
        K_s=self.K_s
        for layer_index in range(self.num_layer):
            Q,K_s=self.decoder_single_layer(Q, K_s, layer_index)
        end = time.time()
        print("decoder.decoder_fn.ended.Q:", Q, " ;K_s:", K_s,";time spent:",(end-start))
        return Q,K_s

    def decoder_single_layer(self,Q,K_s,layer_index):
        """
        singel layer for decoder. each layers has three sub-layers:
        the first is multi-head self-attention(mask) mechanism;
        the second is multi-head attention over output of encoder
        the third is position-wise fully connected feed-forward network.
        for each sublayer. use LayerNorm(x+Sublayer(x)). input and output of last dimension: d_model=512s.
        :param Q: shape should be:       [batch_size,sequence_length,d_model]
        :param K_s: shape should be:     [batch_size,sequence_length,d_model]
        :param layer_index: index of layer
        :param mask: mask is a list. length is sequence_length. each element is a scaler value. e.g. [1,1,1,-1000000,-1000000,-1000000,....-1000000]
        :return:output: shape should be:[batch_size*sequence_length,d_model]
        """
        print("#decoder#decoder_single_layer",layer_index,"====================================>")
        # 1.1 the first is masked multi-head self-attention mechanism
        multi_head_attention_output=self.sub_layer_multi_head_attention(layer_index,Q,K_s,self.type,is_training=self.is_training,mask=self.mask,dropout_keep_prob=self.dropout_keep_prob) #[batch_size*sequence_length,d_model]
        #1.2 use LayerNorm(x+Sublayer(x)). all dimension=512.
        multi_head_attention_output=self.sub_layer_layer_norm_residual_connection(K_s,multi_head_attention_output,layer_index,'decoder_multi_head_attention',dropout_keep_prob=self.dropout_keep_prob)


        # 2.1 the second is multi-head attention over output of encoder
        # IMPORTANT!!! check two parameters below: Q: should from decoder; K_s: should be the output of encoder
        attention_enc_dec=AttentionEncoderDecoder(self.d_model,self.d_k,self.d_v,self.sequence_length,self.h,self.batch_size,
                                                  multi_head_attention_output,self.K_v_encoder,layer_index,self.decoder_sent_length,dropout_keep_prob=self.dropout_keep_prob)
        attention_enc_dec_output=attention_enc_dec.attention_encoder_decoder_fn()
        print("decoder.2.1.attention_enc_dec_output:",attention_enc_dec_output)
        # 2.2 use LayerNorm(x+Sublayer(x)). all dimension=512.
        attention_enc_dec_output=self.sub_layer_layer_norm_residual_connection(multi_head_attention_output,attention_enc_dec_output, layer_index,
                                                                               'decoder_attention_encoder_decoder',dropout_keep_prob=self.dropout_keep_prob)

        # 3.1 the second is position-wise fully connected feed-forward network.
        postion_wise_feed_forward_output=self.sub_layer_postion_wise_feed_forward(attention_enc_dec_output,layer_index,self.type)
        print("decoder.3.1.postion_wise_feed_forward_output:",postion_wise_feed_forward_output)
        #3.2 use LayerNorm(x+Sublayer(x)). all dimension=512.
        postion_wise_feed_forward_output=self.sub_layer_layer_norm_residual_connection(attention_enc_dec_output,postion_wise_feed_forward_output, layer_index, 'decoder_position_ff',dropout_keep_prob=self.dropout_keep_prob)
        return postion_wise_feed_forward_output,postion_wise_feed_forward_output

#####################BELOW IS TEST METHOD FOR DECODER: FROM SMALL FUNCTION  TO WHOLE FUNCTION OF DECODER：################################################
#test decoder for single layer

def get_mask(sequence_length):
    lower_triangle=tf.matrix_band_part(tf.ones([sequence_length,sequence_length]),-1,0)
    result=-1e9*(1.0-lower_triangle)
    print("get_mask==>result:",result)
    return result

def init():
    d_model = 512
    d_k = 64
    d_v = 64
    sequence_length =6 #5
    decoder_sent_length=6
    h = 8
    batch_size = 4*32
    num_layer=6
    # 2.set Q,K,V
    vocab_size = 1000
    embed_size = d_model
    initializer = tf.random_normal_initializer(stddev=0.1)
    Embedding = tf.get_variable("Embedding_d", shape=[vocab_size, embed_size], initializer=initializer)
    decoder_input_x = tf.placeholder(tf.int32, [batch_size, decoder_sent_length], name="input_x")  # [4,10]
    print("1.decoder_input_x:", decoder_input_x)
    decoder_input_embedding = tf.nn.embedding_lookup(Embedding, decoder_input_x)  # [batch_size*sequence_length,embed_size]
    #Q = embedded_words  # [batch_size*sequence_length,embed_size]
    #K_s = embedded_words  # [batch_size*sequence_length,embed_size]
    #K_v_encoder = tf.placeholder(tf.float32, [batch_size,decoder_sent_length, d_model], name="input_x") #sequence_length
    Q = tf.placeholder(tf.float32, [batch_size,sequence_length, d_model], name="input_x")
    K_s=decoder_input_embedding
    K_v_encoder= tf.get_variable("v_variable",shape=[batch_size,decoder_sent_length, d_model],initializer=initializer) #tf.float32,
    print("2.output from encoder:",K_v_encoder)
    mask = get_mask(decoder_sent_length) #sequence_length
    decoder = Decoder(d_model, d_k, d_v, sequence_length, h, batch_size, Q, K_s, K_v_encoder,decoder_sent_length,mask=mask,num_layer=num_layer)
    return decoder,Q, K_s

decoder,Q, K_s=init()

def test_decoder_single_layer():
    layer_index=0
    print("Q.get_shape().as_list():",Q.get_shape().as_list())
    sequence_length_unfold=Q.get_shape().as_list()[0]
    mask=tf.ones((sequence_length_unfold)) #,dtype=tf.float32
    decoder.decoder_single_layer(Q,K_s,layer_index,mask)

def test_decoder():
    output=decoder.decoder_fn()
    return output

#1. test for single layer of decoder
#test_decoder_single_layer()

#2. test for decoder
#decoder_output=test_decoder(); print("3.decoder_output:",decoder_output)


