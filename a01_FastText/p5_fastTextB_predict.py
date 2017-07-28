# -*- coding: utf-8 -*-
#prediction using model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.predict
try:
    reload                        # Python 2
except NameError:
    from importlib import reload  # Python 3
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from p5_fastTextB_model import fastTextB as fastText
from data_util_zhihu import load_data_predict,load_final_test_data,create_voabulary,create_voabulary_label
from tflearn.data_utils import to_categorical, pad_sequences
import os
import codecs

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("label_size",1999,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate.") #批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少
tf.app.flags.DEFINE_integer("num_sampled",100,"number of noise sampling")
tf.app.flags.DEFINE_string("ckpt_dir","fast_text_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",300,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",15,"embedding size")
tf.app.flags.DEFINE_integer("validate_every", 3, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_string("predict_target_file","fast_text_checkpoint/zhihu_result_ftB2.csv","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'test-zhihu-forpredict-v4only-title.txt',"target file path for final prediction")

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    # 1.load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word = create_voabulary()
    vocab_size = len(vocabulary_word2index)
    vocabulary_word2index_label,vocabulary_index2word_label = create_voabulary_label()
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
        fast_text=fastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,FLAGS.num_sampled,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        # 5.feed data, to get logits
        number_of_training_data=len(testX2);print("number_of_training_data:",number_of_training_data)
        batch_size=1
        index=0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data+1, batch_size)):
            logits=sess.run(fast_text.logits,feed_dict={fast_text.sentence:testX2[start:end]}) #'shape of logits:', ( 1, 1999)
            # 6. get lable using logtis
            predicted_labels=get_label_using_logits(logits[0],vocabulary_index2word_label)
            # 7. write question id and labels to file system.
            write_question_id_with_labels(question_id_list[index],predicted_labels,predict_target_file_f)
            index=index+1
        predict_target_file_f.close()

# get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=5):
    # test
    #print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    label_list=[]
    for index in index_list:
        label=vocabulary_index2word_label[index]
        label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return label_list

# write question id and labels to file system.
def write_question_id_with_labels(question_id,labels_list,f):
    labels_string=",".join(labels_list)
    f.write(question_id+","+labels_string+"\n")

if __name__ == "__main__":
    tf.app.run()
