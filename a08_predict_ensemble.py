# -*- coding: utf-8 -*-
#prediction using model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.predict
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
import os
from a3_entity_network import EntityNetwork
sys.path.append("..")
from a08_DynamicMemoryNetwork.data_util_zhihu import load_data_predict,load_final_test_data,create_voabulary,create_voabulary_label
from tflearn.data_utils import pad_sequences #to_categorical
import codecs
from a08_DynamicMemoryNetwork.a8_dynamic_memory_network import DynamicMemoryNetwork
from p7_TextCNN_model import TextCNN
from p71_TextRCNN_mode2 import TextRCNN

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",1999,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 80, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir_dmn","../checkpoint_dynamic_memory_network/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",60,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",1,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
#tf.app.flags.DEFINE_string("cache_path","text_cnn_checkpoint/data_cache.pik","checkpoint location for the model")
tf.app.flags.DEFINE_string("traning_data_path","../train-zhihu4-only-title-all.txt","path of traning data.") #O.K.train-zhihu4-only-title-all.txt-->training-data/test-zhihu4-only-title.txt--->'training-data/train-zhihu5-only-title-multilabel.txt'
tf.app.flags.DEFINE_string("word2vec_model_path","../zhihu-word2vec-title-desc.bin-100","word2vec's vocabulary and vectors") #zhihu-word2vec.bin-100-->zhihu-word2vec-multilabel-minicount15.bin-100
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
tf.app.flags.DEFINE_integer("hidden_size",100,"hidden size")
tf.app.flags.DEFINE_string("predict_target_file","zhihu_result_ensemble_2_0814.csv","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'../test-zhihu-forpredict-title-desc-v6.txt',"target file path for final prediction") #test-zhihu-forpredict-v4only-title.txt
tf.app.flags.DEFINE_integer("story_length",1,"story length")
tf.app.flags.DEFINE_boolean("use_gated_gru",True,"whether to use gated gru as  memory update mechanism. if false,use weighted sum of candidate sentences according to gate")
tf.app.flags.DEFINE_integer("num_pass",3,"number of pass to run") #e.g. num_pass=1,2,3,4.
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")
tf.app.flags.DEFINE_boolean("decode_with_sequences",False,"if your task is sequence generating, you need to set this true.default is false, for predict a label")
###################################above from dynamic memory. below from entityNet#######################################################################################
tf.app.flags.DEFINE_string("ckpt_dir_entity","../checkpoint_entity_network5-b40-60-l2B/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("block_size",40,"block size")
tf.app.flags.DEFINE_boolean("use_bi_lstm",True,"whether to use bi-directional lstm for encode of story and query")
###################################above from dynamic memory. below from entityNet#######################################################################################
tf.app.flags.DEFINE_string("ckpt_dir_cnn","../checkpoint_text_cnn/text_cnn_title_desc_checkpoint_exp512/bak_important/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",100,"max sentence length")
tf.app.flags.DEFINE_integer("num_filters", 512, "number of filters") #128
filter_sizes=[3,4,5,7,10,15,20,25]
###################################above is TextRCNN######################################################################################################################
tf.app.flags.DEFINE_string("ckpt_dir_rcnn","../checkpoint_rcnn/text_rcnn_title_desc_checkpoint2/","checkpoint location for the model")
#tf.app.flags.DEFINE_integer("sentence_len",100,"max sentence length")
###################################above is RCNN############################################################################################################################
###################################above is TextCNN_256embedding############################################################################################################
tf.app.flags.DEFINE_string("ckpt_dir_cnn_256_embedding","../checkpoint_text_cnn/text_cnn_title_desc_checkpoint_exp512_0814/","checkpoint location for the model")
filter_sizes_256_embedding=[3,4,5,6,7,8,9,10,15,20,25] #[1,2,3,4,5,6,7,8,9,10]#[1,2,3,4,5,6,7,8,9]#[5,6,7,8,9] #[2,3,5,6,7,8]#[3,4,5,7,10,15,20,25] #[1,2,3,4,5,6,7][3,5,7]#[7,8,9,10,15,20,25] #[3,4,5,7,10,15,20,25]-->[6,7,8,10,15,20,25,30,35]BAD EPOCH2:13.2  #
tf.app.flags.DEFINE_integer("num_filters_256_embedding", 128, "number of filters") #256--->512--->600
tf.app.flags.DEFINE_integer("embed_size_256_embedding", 256, "embedding and hidden size") #256--->512--->600
###################################above is TextCNN_256embedding############################################################################################################
###################################above is HAN############################################################################################################
tf.app.flags.DEFINE_string("ckpt_dir_cnn_256_embedding","../checkpoint_text_cnn/text_cnn_title_desc_checkpoint_exp512_0814/","checkpoint location for the model")

###################################above is THAN############################################################################################################

def main(_):
    # 1.load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word = create_voabulary(word2vec_model_path=FLAGS.word2vec_model_path,name_scope="dynamic_memory_network")
    vocab_size = len(vocabulary_word2index)
    vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(name_scope="dynamic_memory_network")
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
    testX2 = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
    testX2_cnn = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length, for CNN
    print("end padding...")
   # 3.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    graph1 = tf.Graph().as_default()
    graph2 = tf.Graph().as_default()
    graph3 = tf.Graph().as_default()
    graph4 = tf.Graph().as_default()
    graph5 = tf.Graph().as_default()
    global sess_dmn
    global sess_entity
    global sess_cnn
    global sess_rcnn
    with graph1:#DynamicMemoryNetwork
        sess_dmn = tf.Session(config=config)
        model_dmn = DynamicMemoryNetwork(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length,
                                     FLAGS.story_length,vocab_size, FLAGS.embed_size, FLAGS.hidden_size, FLAGS.is_training,num_pass=FLAGS.num_pass,
                                     use_gated_gru=FLAGS.use_gated_gru,decode_with_sequences=FLAGS.decode_with_sequences,multi_label_flag=FLAGS.multi_label_flag,l2_lambda=FLAGS.l2_lambda)
        saver_dmn = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir_dmn + "checkpoint"):
            print("Restoring Variables from Checkpoint of DMN.")
            saver_dmn.restore(sess_dmn, tf.train.latest_checkpoint(FLAGS.ckpt_dir_dmn))
        else:
            print("Can't find the checkpoint.going to stop.DMN")
            return
    with graph2:#EntityNet
        sess_entity = tf.Session(config=config)
        model_entity = EntityNetwork(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length,
                              FLAGS.story_length,vocab_size, FLAGS.embed_size, FLAGS.hidden_size, FLAGS.is_training,
                              multi_label_flag=True, block_size=FLAGS.block_size,use_bi_lstm=FLAGS.use_bi_lstm)
        saver_entity = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir_entity + "checkpoint"):
            print("Restoring Variables from Checkpoint of EntityNet.")
            saver_entity.restore(sess_entity, tf.train.latest_checkpoint(FLAGS.ckpt_dir_entity))
        else:
            print("Can't find the checkpoint.going to stop.EntityNet.")
            return
    with graph3:#TextCNN
        sess_cnn=tf.Session(config=config)
        model_cnn = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size,
                          FLAGS.decay_steps, FLAGS.decay_rate,FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)
        saver_cnn = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir_cnn + "checkpoint"):
            print("Restoring Variables from Checkpoint.TextCNN.")
            saver_cnn.restore(sess_cnn, tf.train.latest_checkpoint(FLAGS.ckpt_dir_cnn))
        else:
            print("Can't find the checkpoint.going to stop.TextCNN.")
            return
    with graph5:  #TextCNN_256embedding
        sess_cnn_256_embedding = tf.Session(config=config)
        model_cnn_256_embedding = TextCNN(filter_sizes_256_embedding, FLAGS.num_filters_256_embedding, FLAGS.num_classes, FLAGS.learning_rate,
                                FLAGS.batch_size,FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len, vocab_size,
                                FLAGS.embed_size_256_embedding, FLAGS.is_training)
        saver_cnn_256_embedding = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir_cnn_256_embedding + "checkpoint"):
            print("Restoring Variables from Checkpoint.TextCNN_256_embedding")
            saver_cnn_256_embedding.restore(sess_cnn_256_embedding, tf.train.latest_checkpoint(FLAGS.ckpt_dir_cnn_256_embedding))
        else:
            print("Can't find the checkpoint.going to stop.TextCNN_256_embedding.")
            return
    #with graph4:#RCNN
    #    sess_rcnn=tf.Session(config=config)
    #    model_rcnn=TextRCNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.decay_steps, FLAGS.decay_rate,FLAGS.sentence_len,
    #            vocab_size,FLAGS.embed_size,FLAGS.is_training,FLAGS.batch_size,multi_label_flag=FLAGS.multi_label_flag)
    #    saver_rcnn = tf.train.Saver()
    #    if os.path.exists(FLAGS.ckpt_dir_rcnn + "checkpoint"):
    #        print("Restoring Variables from Checkpoint.TextRCNN.")
    #        saver_rcnn.restore(sess_rcnn, tf.train.latest_checkpoint(FLAGS.ckpt_dir_rcnn))
    #    else:
    #        print("Can't find the checkpoint.going to stop.TextRCNN.")
    #        return

        # 5.feed data, to get logits
        number_of_training_data=len(testX2);print("number_of_training_data:",number_of_training_data)
        index=0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        global sess_dmn
        global sess_entity
        for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
            #1.DMN
            logits_dmn=sess_dmn.run(model_dmn.logits,feed_dict={model_dmn.query:testX2[start:end],model_dmn.story: np.expand_dims(testX2[start:end],axis=1),
                                                        model_dmn.dropout_keep_prob:1.0})
            #2.EntityNet
            logits_entity=sess_entity.run(model_entity.logits,feed_dict={model_entity.query:testX2[start:end],model_entity.story: np.expand_dims(testX2[start:end],axis=1),
                                                        model_entity.dropout_keep_prob:1.0})
            #3.CNN
            logits_cnn = sess_cnn.run(model_cnn.logits,feed_dict={model_cnn.input_x: testX2_cnn[start:end], model_cnn.dropout_keep_prob: 1})
            #4.RCNN
            #logits_rcnn = sess_rcnn.run(model_rcnn.logits, feed_dict={model_rcnn.input_x: testX2_cnn[start:end],model_rcnn.dropout_keep_prob: 1})  # 'shape of logits:', ( 1, 1999)
            #5.CN_256_original_embeddding
            logits_cnn_256_embedding =sess_cnn_256_embedding.run(model_cnn_256_embedding.logits,feed_dict={model_cnn_256_embedding.input_x: testX2_cnn[start:end],
                                                                 model_cnn_256_embedding.dropout_keep_prob: 1})
            #how to combine to logits: average
            logits=logits_cnn*0.3+logits_cnn_256_embedding*0.3+logits_entity*0.2+logits_dmn*0.2#+logits_rcnn*0.15
            question_id_sublist=question_id_list[start:end]
            get_label_using_logits_batch(question_id_sublist, logits, vocabulary_index2word_label, predict_target_file_f)
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

# get label using logits
def get_label_using_logits_batch(question_id_sublist,logits_batch,vocabulary_index2word_label,f,top_number=5):
    #print("get_label_using_logits.shape:", logits_batch.shape) # (10, 1999))=[batch_size,num_labels]===>需要(10,5)
    for i,logits in enumerate(logits_batch):
        index_list=np.argsort(logits)[-top_number:] #print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
        index_list=index_list[::-1]
        label_list=[]
        for index in index_list:
            label=vocabulary_index2word_label[index]
            label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
        #print("get_label_using_logits.label_list",label_list)
        write_question_id_with_labels(question_id_sublist[i], label_list, f)
    f.flush()
    #return label_list
# write question id and labels to file system.
def write_question_id_with_labels(question_id,labels_list,f):
    labels_string=",".join(labels_list)
    f.write(question_id+","+labels_string+"\n")

if __name__ == "__main__":
    tf.app.run()