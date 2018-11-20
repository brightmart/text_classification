# coding=utf-8
"""
train bert model
"""
import modeling
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Describe your program')
parser.add_argument('-batch_size', '--batch_size', type=int,default=128)
args = parser.parse_args()
batch_size=args.batch_size
print("batch_size:",batch_size)
def bert_train_fn():
    is_training=True
    hidden_size = 768
    num_labels = 10
    #batch_size=128
    max_seq_length=512
    use_one_hot_embeddings = False
    bert_config = modeling.BertConfig(vocab_size=21128, hidden_size=hidden_size, num_hidden_layers=12,
                                      num_attention_heads=12,intermediate_size=3072)

    input_ids = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_ids")
    input_mask = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_mask")
    segment_ids = tf.placeholder(tf.int32, [batch_size,max_seq_length],name="segment_ids")
    label_ids = tf.placeholder(tf.float32, [batch_size,num_labels], name="label_ids")
    loss, per_example_loss, logits, probabilities, model = create_model(bert_config, is_training, input_ids, input_mask,
                                                                        segment_ids, label_ids, num_labels,
                                                                        use_one_hot_embeddings)
    # 1. generate or load training/validation/test data. e.g. train:(X,y). X is input_ids,y is labels.

    # 2. train the model by calling create model, get loss
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        input_ids_=np.ones((batch_size,max_seq_length),dtype=np.int32)
        input_mask_=np.ones((batch_size,max_seq_length),dtype=np.int32)
        segment_ids_=np.ones((batch_size,max_seq_length),dtype=np.int32)
        label_ids_=np.ones((batch_size,num_labels),dtype=np.float32)
        feed_dict = {input_ids: input_ids_, input_mask: input_mask_,segment_ids:segment_ids_,label_ids:label_ids_}
        loss_ = sess.run([loss], feed_dict)
        print("loss:",loss_)
    # 3. eval the model from time to time

def bert_predict_fn():
    # 1. predict based on
    pass

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()
  hidden_size = output_layer.shape[-1].value
  output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
  output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:  # if training, add dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    print("output_layer:",output_layer.shape,";output_weights:",output_weights.shape,";logits:",logits.shape)

    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    per_example_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(per_example_loss)

    return loss, per_example_loss, logits, probabilities,model

bert_train_fn()
