# -*- coding: utf-8 -*-
#prediction using model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.predict
# currently this file is not well test, so you can just ignore this file util it is tested or write a function, input is
# a strings,output is predicted labels.
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
#from p5_fastTextB_model import fastTextB as fastText
#from a02_TextCNN.other_experiement.data_util_zhihu import load_data_predict,load_final_test_data,create_voabulary,create_voabulary_label
from tflearn.data_utils import pad_sequences #to_categorical
import os
import codecs
from p7_TextCNN_model import TextCNN
import pickle
import h5py
from data_util import load_data
#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate.") #批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","text_cnn_title_desc_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",200,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",15,"number of epochs.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_string("predict_target_file","text_cnn_title_desc_checkpoint/zhihu_result_cnn_multilabel_v6_e14.csv","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'test-zhihu-forpredict-title-desc-v6.txt',"target file path for final prediction") #test-zhihu-forpredict-v4only-title.txt
tf.app.flags.DEFINE_string("word2vec_model_path","zhihu-word2vec-title-desc.bin-100","word2vec's vocabulary and vectors") #zhihu-word2vec.bin-100
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters") #128

##############################################################################################################################################
filter_sizes=[6,7,8]#[1,2,3,4,5,6,7]
#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
# 1.load data with vocabulary of words and labels
#vocabulary_word2index, vocabulary_index2word = create_voabulary(simple='simple',
#                                                                word2vec_model_path=FLAGS.word2vec_model_path,name_scope="cnn2")
word2index, label2index, _, _, _, _, _, _ = load_data(FLAGS.cache_file_h5py,FLAGS.cache_file_pickle)

vocab_size = len(word2index)
vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(name_scope="cnn2")
questionid_question_lists = load_final_test_data(FLAGS.predict_source_file)
test = load_data_predict(word2index, vocabulary_word2index_label, questionid_question_lists)
testX = []
question_id_list = []
for tuple in test:
    question_id, question_string_list = tuple
    question_id_list.append(question_id)
    testX.append(question_string_list)
# 2.Data preprocessing: Sequence padding
print("start padding....")
testX2 = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
print("end padding...")
# 3.create session.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph=tf.Graph().as_default()
global sess
global textCNN
with graph:
    sess=tf.Session(config=config)
# 4.Instantiate Model
    textCNN = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size,
                  FLAGS.decay_steps, FLAGS.decay_rate,
                  FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)
    saver = tf.train.Saver()
    if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
        print("Restoring Variables from Checkpoint")
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
    else:
        print("Can't find the checkpoint.going to stop")
    #return
# 5.feed data, to get logits
number_of_training_data = len(testX2);
print("number_of_training_data:", number_of_training_data)
#index = 0
#predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
#############################################################################################################################################
def get_logits_with_value_by_input(start,end):
    x=testX2[start:end]
    global sess
    global textCNN
    logits = sess.run(textCNN.logits, feed_dict={textCNN.input_x: x, textCNN.dropout_keep_prob: 1})
    predicted_labels,value_labels = get_label_using_logits_with_value(logits[0], vocabulary_index2word_label)
    value_labels_exp= np.exp(value_labels)
    p_labels=value_labels_exp/np.sum(value_labels_exp)
    return predicted_labels,p_labels

def main(_):
    # 1.load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word = create_voabulary(simple='simple',word2vec_model_path=FLAGS.word2vec_model_path,name_scope="cnn2")
    vocab_size = len(vocabulary_word2index)
    vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(name_scope="cnn2")
    questionid_question_lists=load_final_test_data(FLAGS.predict_source_file)
    test= load_data_predict(vocabulary_word2index,vocabulary_word2index_label,questionid_question_lists)
    testX=[]
    question_id_list=[]
    for tuple in test:
        question_id,question_string_list=tuple
        question_id_list.append(question_id)
        testX.append(question_string_list)
    # 2.Data preprocessing: Sequence padding
    print("start padding....")
    testX2 = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    print("end padding...")
   # 3.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # 4.Instantiate Model
        textCNN=TextCNN(filter_sizes,FLAGS.num_filters,FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,FLAGS.decay_rate,
                        FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        # 5.feed data, to get logits
        number_of_training_data=len(testX2);print("number_of_training_data:",number_of_training_data)
        index=0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
            logits=sess.run(textCNN.logits,feed_dict={textCNN.input_x:testX2[start:end],textCNN.dropout_keep_prob:1}) #'shape of logits:', ( 1, 1999)
            # 6. get lable using logtis
            predicted_labels=get_label_using_logits(logits[0],vocabulary_index2word_label)
            # 7. write question id and labels to file system.
            write_question_id_with_labels(question_id_list[index],predicted_labels,predict_target_file_f)
            index=index+1
        predict_target_file_f.close()

# get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=5):
    index_list=np.argsort(logits)[-top_number:] #print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list=index_list[::-1]
    label_list=[]
    for index in index_list:
        label=vocabulary_index2word_label[index]
        label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return label_list

# get label using logits
def get_label_using_logits_with_value(logits,vocabulary_index2word_label,top_number=5):
    index_list=np.argsort(logits)[-top_number:] #print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list=index_list[::-1]
    value_list=[]
    label_list=[]
    for index in index_list:
        label=vocabulary_index2word_label[index]
        label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
        value_list.append(logits[index])
    return label_list,value_list

# write question id and labels to file system.
def write_question_id_with_labels(question_id,labels_list,f):
    labels_string=",".join(labels_list)
    f.write(question_id+","+labels_string+"\n")

def load_data(cache_file_h5py,cache_file_pickle):
    """
    load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download cache file, it include training data and vocabulary & labels. "
                           "link can be found in README.md\n download zip file, unzip it, then put cache files as FLAGS."
                           "cache_file_h5py and FLAGS.cache_file_pickle suggested location.")
    print("INFO. cache file exists. going to load cache file")
    f_data = h5py.File(cache_file_h5py, 'r')
    print("f_data.keys:",list(f_data.keys()))
    train_X=f_data['train_X'] # np.array(
    print("train_X.shape:",train_X.shape)
    train_Y=f_data['train_Y'] # np.array(
    print("train_Y.shape:",train_Y.shape,";")
    vaild_X=f_data['vaild_X'] # np.array(
    valid_Y=f_data['valid_Y'] # np.array(
    test_X=f_data['test_X'] # np.array(
    test_Y=f_data['test_Y'] # np.array(
    #f_data.close()

    word2index, label2index=None,None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index=pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    return word2index, label2index,train_X,train_Y,vaild_X,valid_Y,test_X,test_Y

if __name__ == "__main__":
    #tf.app.run()
    labels,list_value=get_logits_with_value_by_input(0, 1)
    print("labels:",labels)
    print("list_value:", list_value)