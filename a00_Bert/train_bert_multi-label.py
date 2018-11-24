# coding=utf-8
"""
train bert model

1.get training data and vocabulary & labels dict
2. create model
3. train the model and report f1 score
"""
import bert_modeling as modeling
import tensorflow as tf
import os
import numpy as np

from utils import load_data,init_label_dict,get_target_label_short,compute_confuse_matrix,compute_micro_macro,compute_confuse_matrix_batch,get_label_using_logits_batch,get_target_label_short_batch

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("cache_file_h5py","../data/ieee_zhihu_cup/data.h5","path of training/validation/test data.") #../data/sample_multiple_label.txt
tf.app.flags.DEFINE_string("cache_file_pickle","../data/ieee_zhihu_cup/vocab_label.pik","path of vocabulary and label files") #../data/sample_multiple_label.txt

tf.app.flags.DEFINE_float("learning_rate",0.0001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_boolean("is_training",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",15,"number of epochs to run.")

# below hyper-parameter is for bert model
# to train a big model,                     use hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072
# to train a middel size model, train fast. use hidden_size=128, num_hidden_layers=4, num_attention_heads=8, intermediate_size=1024
tf.app.flags.DEFINE_integer("hidden_size",128,"hidden size") # 768
tf.app.flags.DEFINE_integer("num_hidden_layers",2,"number of hidden layers") # 12--->4
tf.app.flags.DEFINE_integer("num_attention_heads",4,"number of attention headers") # 12
tf.app.flags.DEFINE_integer("intermediate_size",256,"intermediate size of hidden layer") # 3072-->512
tf.app.flags.DEFINE_integer("max_seq_length",200,"max sequence length")

def main(_):
    # 1. get training data and vocabulary & labels dict
    word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY = load_data(FLAGS.cache_file_h5py,FLAGS.cache_file_pickle)
    vocab_size = len(word2index); print("bert model.vocab_size:", vocab_size);
    num_labels = len(label2index); print("num_labels:", num_labels); cls_id=word2index['CLS'];print("id of 'CLS':",word2index['CLS'])
    num_examples, FLAGS.max_seq_length = trainX.shape;print("num_examples of training:", num_examples, ";max_seq_length:", FLAGS.max_seq_length)

    # 2. create model, define train operation
    bert_config = modeling.BertConfig(vocab_size=len(word2index), hidden_size=FLAGS.hidden_size, num_hidden_layers=FLAGS.num_hidden_layers,
                                      num_attention_heads=FLAGS.num_attention_heads,intermediate_size=FLAGS.intermediate_size)
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_ids") # FLAGS.batch_size
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_mask")
    segment_ids = tf.placeholder(tf.int32, [None,FLAGS.max_seq_length],name="segment_ids")
    label_ids = tf.placeholder(tf.float32, [None,num_labels], name="label_ids")
    is_training = tf.placeholder(tf.bool, name="is_training") # FLAGS.is_training

    use_one_hot_embeddings = False
    loss, per_example_loss, logits, probabilities, model = create_model(bert_config, is_training, input_ids, input_mask,
                                                            segment_ids, label_ids, num_labels,use_one_hot_embeddings)
    # define train operation
    #num_train_steps = int(float(num_examples) / float(FLAGS.batch_size * FLAGS.num_epochs)); use_tpu=False; num_warmup_steps = int(num_train_steps * 0.1)
    #train_op = optimization.create_optimizer(loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, use_tpu)
    global_step = tf.Variable(0, trainable=False, name="Global_Step")
    train_op = tf.contrib.layers.optimize_loss(loss, global_step=global_step, learning_rate=FLAGS.learning_rate,optimizer="Adam", clip_gradients=3.0)

    # 3. train the model by calling create model, get loss
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
        print("Checkpoint Exists. Restoring Variables from Checkpoint.")
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
    number_of_training_data = len(trainX)
    iteration = 0
    curr_epoch = 0 #sess.run(textCNN.epoch_step)
    batch_size = FLAGS.batch_size
    for epoch in range(curr_epoch, FLAGS.num_epochs):
        loss_total, counter = 0.0, 0
        for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
            iteration = iteration + 1 ###
            input_mask_, segment_ids_, input_ids_=get_input_mask_segment_ids(trainX[start:end],cls_id) # input_ids_,input_mask_,segment_ids_
            feed_dict = {input_ids: input_ids_, input_mask: input_mask_, segment_ids:segment_ids_,
                         label_ids:trainY[start:end],is_training:True}
            curr_loss,_ = sess.run([loss,train_op], feed_dict)
            loss_total, counter = loss_total + curr_loss, counter + 1
            if counter % 30 == 0:
                print(epoch,"\t",iteration,"\tloss:",loss_total/float(counter),"\tcurrent_loss:",curr_loss)
            if counter % 300==0:
                print("input_ids[",start,"]:",input_ids_[0]);#print("trainY[start:end]:",trainY[start:end])
                try:
                    target_labels = get_target_label_short_batch(trainY[start:end]);#print("target_labels:",target_labels)
                    print("trainY[",start,"]:",target_labels[0])
                except:
                    pass
            # evaulation
            if start!=0 and start % (1000 * FLAGS.batch_size) == 0:
                eval_loss, f1_score, f1_micro, f1_macro = do_eval(sess,input_ids,input_mask,segment_ids,label_ids,is_training,loss,
                                                                  probabilities,vaildX, vaildY, num_labels,batch_size,cls_id)
                print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (
                    epoch, eval_loss, f1_score, f1_micro, f1_macro))
                # save model to checkpoint
                #if start % (4000 * FLAGS.batch_size)==0:
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                print("Going to save model..")
                saver.save(sess, save_path, global_step=epoch)

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,labels, num_labels, use_one_hot_embeddings,reuse_flag=False):
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
  with tf.variable_scope("weights",reuse=reuse_flag):
      output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
      output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    #if is_training:
    #    print("###create_model.is_training:",is_training)
    #    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    def apply_dropout_last_layer(output_layer):
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        return output_layer

    def not_apply_dropout(output_layer):
        return output_layer

    output_layer=tf.cond(is_training, lambda: apply_dropout_last_layer(output_layer), lambda:not_apply_dropout(output_layer))
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    print("output_layer:",output_layer.shape,";output_weights:",output_weights.shape,";logits:",logits.shape) # shape=(?, 1999)

    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.sigmoid(logits) #tf.nn.softmax(logits, axis=-1)
    per_example_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits) # shape=(?, 1999)
    loss_batch = tf.reduce_sum(per_example_loss,axis=1)  #  (?,)
    loss=tf.reduce_mean(loss_batch) #  (?,)

    return loss, per_example_loss, logits, probabilities,model


def do_eval(sess,input_ids,input_mask,segment_ids,label_ids,is_training,loss,probabilities,vaildX, vaildY, num_labels,batch_size,cls_id):
    """
    evalution on model using validation data
    """
    num_eval=1000
    vaildX = vaildX[0:num_eval]
    vaildY = vaildY[0:num_eval]
    number_examples = len(vaildX)
    eval_loss, eval_counter, eval_f1_score, eval_p, eval_r = 0.0, 0, 0.0, 0.0, 0.0
    label_dict = init_label_dict(num_labels)
    print("do_eval.number_examples:",number_examples)
    f1_score_micro_sklearn_total=0.0
    # batch_size=1 # TODO
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        input_mask_, segment_ids_, input_ids_ = get_input_mask_segment_ids(vaildX[start:end],cls_id)
        feed_dict = {input_ids: input_ids_,input_mask:input_mask_,segment_ids:segment_ids_,
                     label_ids:vaildY[start:end],is_training:False}
        curr_eval_loss, prob = sess.run([loss, probabilities],feed_dict)
        target_labels=get_target_label_short_batch(vaildY[start:end])
        predict_labels=get_label_using_logits_batch(prob)
        if start%100==0:
            print("prob.shape:",prob.shape,";prob:",prob)
            print("predict_labels:",predict_labels)

        #print("predict_labels:",predict_labels)
        label_dict=compute_confuse_matrix_batch(target_labels,predict_labels,label_dict,name='bert')
        eval_loss, eval_counter = eval_loss + curr_eval_loss, eval_counter + 1

    f1_micro, f1_macro = compute_micro_macro(label_dict)  # label_dictis a dict, key is: accusation,value is: (TP,FP,FN). where TP is number of True Positive
    f1_score_result = (f1_micro + f1_macro) / 2.0
    return eval_loss / float(eval_counter+0.00001), f1_score_result, f1_micro, f1_macro

def get_input_mask_segment_ids(train_x_batch,cls_id):
    """
    get input mask and segment ids given a batch of input x.
    if sequence length of input x is max_sequence_length, then shape of both input_mask and segment_ids should be
    [batch_size, max_sequence_length]. for those padding tokens, input_mask will be zero, value for all other place is one.
    :param train_x_batch:
    :return: input_mask_,segment_ids
    """
    batch_size,max_sequence_length=train_x_batch.shape
    input_mask=np.ones((batch_size,max_sequence_length),dtype=np.int32)
    # set 0 for token in padding postion
    for i in range(batch_size):
        input_x_=train_x_batch[i] # a list, length is max_sequence_length
        input_x=list(input_x_)
        for j in range(len(input_x)):
            if input_x[j]==0:
                input_mask[i][j:]=0
                break
    # insert CLS token for classification
    input_ids=np.zeros((batch_size,max_sequence_length),dtype=np.int32)
    #print("input_ids.shape1:",input_ids.shape)
    for k in range(batch_size):
        input_id_list=list(train_x_batch[k])
        input_id_list.insert(0,cls_id)
        del input_id_list[-1]
        input_ids[k]=input_id_list
    #print("input_ids.shape2:",input_ids.shape)

    segment_ids=np.ones((batch_size,max_sequence_length),dtype=np.int32)
    return input_mask, segment_ids,input_ids

#train_x_batch=np.ones((3,5))
#train_x_batch[0,4]=0
#train_x_batch[1,3]=0
#train_x_batch[1,4]=0
#cls_id=2
#print("train_x_batch:",train_x_batch)
#input_mask, segment_ids,input_ids=get_input_mask_segment_ids(train_x_batch,cls_id)
#print("input_mask:",input_mask, "segment_ids:",segment_ids,"input_ids:",input_ids)

if __name__ == "__main__":
    tf.app.run()