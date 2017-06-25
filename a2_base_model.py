import tensorflow as tf
from  a2_multi_head_attention import MultiHeadAttention
from a2_poistion_wise_feed_forward import PositionWiseFeedFoward

class BaseClass(object):
    """
    base class has some common fields and functions.
    """
    def __init__(self,d_model,d_k,d_v,sequence_length,h,batch_size,num_layer=6,type='encoder'):
        """
        :param d_model:
        :param d_k:
        :param d_v:
        :param sequence_length:
        :param h:
        :param batch_size:
        :param embedded_words: shape:[batch_size*sequence_length,embed_size]
        """
        self.d_model=d_model
        self.d_k=d_k
        self.d_v=d_v
        self.sequence_length=sequence_length
        self.h=h
        self.num_layer=num_layer
        self.batch_size=batch_size

    def sub_layer_postion_wise_feed_forward(self ,x ,layer_index,type)  :# COMMON FUNCTION
        """
        :param x: shape should be:[?,d_model]
        :param layer_index: index of layer number
        :param type: encoder,decoder or encoder_decoder_attention
        :return: [?,d_model]
        """
        with tf.variable_scope("sub_layer_postion_wise_feed_forward" + type + str(layer_index)):
            postion_wise_feed_forward = PositionWiseFeedFoward(x, layer_index)
            postion_wise_feed_forward_output = postion_wise_feed_forward.position_wise_feed_forward_fn()
        return postion_wise_feed_forward_output

    def sub_layer_multi_head_attention(self ,layer_index ,Q ,K_s,type,mask=None)  :# COMMON FUNCTION
        """
        multi head attention as sub layer
        :param layer_index: index of layer number
        :param Q: shape should be: [batch_size*sequence_length,embed_size]
        :param k_s: shape should be: [batch_size*sequence_length,embed_size]
        :param type: encoder,decoder or encoder_decoder_attention
        :param use_mask:whether to use mask or not.if use mask,illegal connection will be mask as huge big negative value.so it's possiblitity will become zero.
        input of this layer's shape:[batch_size,sequence_length]
        :return: output of multi head attention.shape:[batch_size,sequence_length,d_model]
        """
        print("===================================>base_model.sub_layer_multi_head_attention.type:",type)
        with tf.variable_scope("base_model_sub_layer_multi_head_attention_" + type+str(layer_index)):
            # below is to handle attention for encoder and decoder with difference length:
            length=self.decoder_sent_length if (type!='encoder' and self.sequence_length!=self.decoder_sent_length) else self.sequence_length
            #1. get V as learned parameters
            V_s = tf.get_variable("V_s", shape=(self.batch_size*length,self.d_model),initializer=self.initializer) ##TODO variable' shape should not include batch_size.
            #2. call function of multi head attention to get result
            multi_head_attention_class = MultiHeadAttention(Q, K_s, V_s, self.d_model, self.d_k, self.d_v, self.sequence_length, self.h,mask=mask)
            sub_layer_multi_head_attention_output = multi_head_attention_class.multi_head_attention_fn()  # [batch_size*sequence_length,d_model]
        return sub_layer_multi_head_attention_output  # [batch_size*sequence_length,d_model]

    def sub_layer_layer_norm_residual_connection(self ,layer_input ,layer_output): # COMMON FUNCTION
        """
        layer norm & residual connection
        :param input: [batch_size*sequence_length,d_model]
        :param output:[batch_size*sequence_length,d_model]
        :return:
        """
        # 1. layer norm
        # TODO
        # 2. residual connection
        assert layer_input.get_shape().as_list()==layer_output.get_shape().as_list()
        layer_output_new= layer_input+ layer_output
        return layer_output_new  # [batch_size,sequence_length,d_model]