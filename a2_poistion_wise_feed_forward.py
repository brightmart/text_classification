
import tensorflow as tf
class PositionWiseFeedFoward(object): #TODO make it parallel
    """
    position-wise feed forward networks. formula as below:
    FFN(x)=max(0,xW1+b1)W2+b2
    """
    def __init__(self,x,layer_index,d_model=512,d_ff=2048):
        """
        :param x: shape should be:[?,d_model]
        :param layer_index:  index of layer
        :return: shape:[?,d_model]
        """
        shape_list=x.get_shape().as_list()
        assert(len(shape_list)==2)
        self.length,_=shape_list
        self.x_original=x
        x_=tf.split(x,self.length,axis=0) #list.length is <length>.each element has shape:[1,d_model]
        self.x=x_

        self.layer_index=layer_index
        self.d_model=d_model
        self.d_ff=d_ff
        self.initializer = tf.random_normal_initializer(stddev=0.1)

    def position_wise_feed_forward_fn(self):
        result_list=[]
        for pos in range(self.length):
            result_pos=self.position_wise_feed_forward_fn_single_pos(pos)
            result_list.append(result_pos) #a list. length is <length>. each element is:[1,d_model]
        result=tf.concat(result_list,axis=0) #[length,d_model]
        assert result.get_shape().as_list()==self.x_original.get_shape().as_list()
        return result

    # FFN(x)=max(0,xW1+b1)W2+b2
    def position_wise_feed_forward_fn_single_pos(self,pos):
        with tf.variable_scope("postion_wise_feed_forward"+str(self.layer_index)):
            if pos>0:
                tf.get_variable_scope().reuse_variables()
            W1 = tf.get_variable("W1",shape=[self.d_model,self.d_ff],initializer=self.initializer)
            b1=tf.get_variable("b1",shape=[self.d_ff]) #x is a list.length is <length>.each element has shape:[1,d_model]
            ffn_layer1=tf.nn.relu(tf.matmul(self.x[pos],W1)+b1) #shape:[1,self.d_ff]<=====matmul([1,self.d_model],[self.d_model,self.d_ff])
            W2 = tf.get_variable("W2",shape=[self.d_ff,self.d_model],initializer=self.initializer)
            b2=tf.get_variable("b2",shape=[self.d_model])
            output=tf.matmul(ffn_layer1,W2)+b2 #[1,d_model]<=====matmul([1,self.d_ff],[self.d_ff,self.d_model])
        return output

#test function of position_wise_feed_forward_fn
def test_position_wise_feed_forward_fn():
    x=tf.ones((8*10,512)) #batch_size=8,sequence_length=10
    layer_index=0
    postion_wise_feed_forward=PositionWiseFeedFoward(x,layer_index)
    output=postion_wise_feed_forward.position_wise_feed_forward_fn()
    print("x:",x,";output:",output)
    return output

#with tf.Session() as sess:
#    result=test_position_wise_feed_forward_fn()
#    sess.run(tf.global_variables_initializer())
#    result_=sess.run(result)
#    print("result_:",result_)

