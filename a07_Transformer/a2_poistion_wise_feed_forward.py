# -*- coding: utf-8 -*-
import tensorflow as tf
import time
"""
Position-wise Feed-Forward Networks
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
connected feed-forward network, which is applied to each position separately and identically. This
consists of two linear transformations with a ReLU activation in between.

FFN(x) = max(0,xW1+b1)W2+b2

While the linear transformations are the same across different positions, they use different parameters
from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
The dimensionality of input and output is d_model= 512, and the inner-layer has dimensionalityd_ff= 2048.
"""
class PositionWiseFeedFoward(object): #TODO make it parallel
    """
    position-wise feed forward networks. formula as below:
    FFN(x)=max(0,xW1+b1)W2+b2
    """
    def __init__(self,x,layer_index,d_model=512,d_ff=2048):
        """
        :param x: shape should be:[batch,sequence_length,d_model]
        :param layer_index:  index of layer
        :return: shape:[sequence_length,d_model]
        """
        shape_list=x.get_shape().as_list()
        assert(len(shape_list)==3)
        self.x=x
        self.layer_index=layer_index
        self.d_model=d_model
        self.d_ff=d_ff
        self.initializer = tf.random_normal_initializer(stddev=0.1)

    def position_wise_feed_forward_fn(self):
        """
        x:       [batch,sequence_length,d_model]
        :return: [batch,sequence_length,d_model]
        """
        output=None
        #1.conv1
        input=tf.expand_dims(self.x,axis=3) #[batch,sequence_length,d_model,1]
        # conv2d.input:       [None,sentence_length,embed_size,1]. filter=[filter_size,self.embed_size,1,self.num_filters]
        # output with padding:[None,sentence_length,1,1]
        filter1 = tf.get_variable("filter1"+str(self.layer_index) , shape=[1, self.d_model, 1, 1],initializer=self.initializer)
        ouput_conv1=tf.nn.conv2d(input,filter1,strides=[1,1,1,1],padding="VALID",name="conv1") #[batch,sequence_length,1,1]
        print("output_conv1:",ouput_conv1)

        #2.conv2
        filter2 = tf.get_variable("filter2"+str(self.layer_index), [1, 1, 1, self.d_model], initializer=self.initializer)
        output_conv2=tf.nn.conv2d(ouput_conv1,filter2,strides=[1,1,1,1],padding="VALID",name="conv2") #[batch,sequence_length,1,d_model]
        output=tf.squeeze(output_conv2) #[batch,sequence_length,d_model]
        return output #[batch,sequence_length,d_model]

#test function of position_wise_feed_forward_fn
#time spent:OLD VERSION: length=8000,time spent:35.6s; NEW VERSION:0.03s
def test_position_wise_feed_forward_fn():
    start=time.time()
    x=tf.ones((8,1000,512)) #batch_size=8,sequence_length=10 ;
    layer_index=0
    postion_wise_feed_forward=PositionWiseFeedFoward(x,layer_index)
    output=postion_wise_feed_forward.position_wise_feed_forward_fn()
    end=time.time()
    print("x:",x,";output:",output,";time spent:",(end-start))
    return output

def test():
    with tf.Session() as sess:
        result=test_position_wise_feed_forward_fn()
        sess.run(tf.global_variables_initializer())
        result_=sess.run(result)
        print("result_:",result_)
#test()

