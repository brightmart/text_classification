# -*- coding: utf-8 -*-
import tensorflow as tf

# 【该方法测试的时候使用】返回一个方法。这个方法根据输入的值，得到对应的索引，再得到这个词的embedding.
def extract_argmax_and_embed(embedding, output_projection=None):
    """
    Get a loop_function that extracts the previous symbol and embeds it. Used by decoder.
    :param embedding: embedding tensor for symbol
    :param output_projection: None or a pair (W, B). If provided, each fed previous output will
    first be multiplied by W and added B.
    :return: A loop function
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.matmul(prev, output_projection[0]) + output_projection[1]
        prev_symbol = tf.argmax(prev, 1) #得到对应的INDEX
        emb_prev = tf.gather(embedding, prev_symbol) #得到这个INDEX对应的embedding
        return emb_prev
    return loop_function

# RNN的解码部分。
# 如果是训练，使用训练数据的输入；如果是test,将t时刻的输出作为t+1时刻的s输入
def rnn_decoder_with_attention(decoder_inputs, initial_state, cell, loop_function,attention_states,scope=None):#3D Tensor [batch_size x attn_length x attn_size]
    """RNN decoder for the sequence-to-sequence model.
    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].it is decoder input.
        initial_state: 2D Tensor with shape [batch_size x cell.state_size].it is the encoded vector of input sentences, which represent 'thought vector'
        cell: core_rnn_cell.RNNCell defining the cell function and size.
        loop_function: If not None, this function will be applied to the i-th output
            in order to generate the i+1-st input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].it is represent input X.
        scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
    Returns:
        A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
            shape [batch_size x output_size] containing generated outputs.
        state: The state of each cell at the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].
            (Note that in some cases, like basic RNN cell or GRU cell, outputs and
            states can be the same. They are different for LSTM cells though.)
    """
    with tf.variable_scope(scope or "rnn_decoder"):
        print("rnn_decoder_with_attention started...")
        state = initial_state  #[batch_size x cell.state_size].
        _, hidden_size = state.get_shape().as_list() #200
        attention_states_original=attention_states
        batch_size,sequence_length,_=attention_states.get_shape().as_list()
        outputs = []
        prev = None
        #################################################
        for i, inp in enumerate(decoder_inputs):#循环解码部分的输入。如sentence_length个[batch_size x input_size]
            # 如果是训练，使用训练数据的输入；如果是test, 将t时刻的输出作为t + 1 时刻的s输入
            if loop_function is not None and prev is not None:#测试的时候：如果loop_function不为空且前一个词的值不为空，那么使用前一个的值作为RNN的输入
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            ##ATTENTION#################################################################################################################################################
            # 1.get logits of attention for each encoder input. attention_states:[batch_size x attn_length x attn_size]; query=state:[batch_size x cell.state_size]
            query=state
            W_a = tf.get_variable("W_a", shape=[hidden_size, hidden_size],initializer=tf.random_normal_initializer(stddev=0.1))
            query=tf.matmul(query, W_a) #[batch_size,hidden_size]
            query=tf.expand_dims(query,axis=1) #[batch_size, 1, hidden_size]
            U_a = tf.get_variable("U_a", shape=[hidden_size, hidden_size],initializer=tf.random_normal_initializer(stddev=0.1))
            U_aa = tf.get_variable("U_aa", shape=[ hidden_size])
            attention_states=tf.reshape(attention_states,shape=(-1,hidden_size)) #[batch_size*sentence_length,hidden_size]
            attention_states=tf.matmul(attention_states, U_a) #[batch_size*sentence_length,hidden_size]
            #print("batch_size",batch_size," ;sequence_length:",sequence_length," ;hidden_size:",hidden_size) #print("attention_states:", attention_states) #(?, 200)
            attention_states=tf.reshape(attention_states,shape=(-1,sequence_length,hidden_size)) # TODO [batch_size,sentence_length,hidden_size]
            #query_expanded:            [batch_size,1,             hidden_size]
            #attention_states_reshaped: [batch_size,sentence_length,hidden_size]
            attention_logits=tf.nn.tanh(query+attention_states+U_aa) #[batch_size,sentence_length,hidden_size]. additive style

            # 2.get possibility of attention
            attention_logits=tf.reshape(attention_logits,shape=(-1,hidden_size)) #batch_size*sequence_length [batch_size*sentence_length,hidden_size]
            V_a = tf.get_variable("V_a", shape=[hidden_size,1],initializer=tf.random_normal_initializer(stddev=0.1)) #[hidden_size,1]
            attention_logits=tf.matmul(attention_logits,V_a) #最终需要的是[batch_size*sentence_length,1]<-----[batch_size*sentence_length,hidden_size],[hidden_size,1]
            attention_logits=tf.reshape(attention_logits,shape=(-1,sequence_length)) #attention_logits:[batch_size,sequence_length]
            ##########################################################################################################################################################
            #attention_logits=tf.reduce_sum(attention_logits,2)        #[batch_size x attn_length]
            attention_logits_max=tf.reduce_max(attention_logits,axis=1,keep_dims=True) #[batch_size x 1]
            # possibility distribution for each encoder input.it means how much attention or focus for each encoder input
            p_attention=tf.nn.softmax(attention_logits-attention_logits_max)#[batch_size x attn_length]

            # 3.get weighted sum of hidden state for each encoder input as attention state
            p_attention=tf.expand_dims(p_attention,axis=2)            #[batch_size x attn_length x 1]
            # attention_states:[batch_size x attn_length x attn_size]; p_attention:[batch_size x attn_length];
            attention_final=tf.multiply(attention_states_original,p_attention) #[batch_size x attn_length x attn_size]
            context_vector=tf.reduce_sum(attention_final,axis=1)     #[batch_size x attn_size]
            ############################################################################################################################################################
            #inp:[batch_size x input_size].it is decoder input;  attention_final:[batch_size x attn_size]
            output, state = cell(inp, state,context_vector)          #attention_final TODO 使用RNN走一步
            outputs.append(output) # 将输出添加到结果列表中
            if loop_function is not None:
                prev = output
    print("rnn_decoder_with_attention ended...")
    return outputs, state