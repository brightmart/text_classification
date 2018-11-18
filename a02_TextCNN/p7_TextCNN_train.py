# -*- coding: utf-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') #gb2312
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from p7_TextCNN_model import TextCNN
#from data_util import create_vocabulary,load_data_multilabel
import pickle
import h5py
import os
import random
#configuration
FLAGS=tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string("traning_data_path","../data/sample_multiple_label.txt","path of traning data.") #../data/sample_multiple_label.txt
#tf.app.flags.DEFINE_integer("vocab_size",100000,"maximum vocab size.")

tf.app.flags.DEFINE_string("cache_file_h5py","../data/ieee_zhihu_cup/data.h5","path of training/validation/test data.") #../data/sample_multiple_label.txt
tf.app.flags.DEFINE_string("cache_file_pickle","../data/ieee_zhihu_cup/vocab_label.pik","path of vocabulary and label files") #../data/sample_multiple_label.txt

tf.app.flags.DEFINE_float("learning_rate",0.0003,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","text_cnn_title_desc_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",200,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.app.flags.DEFINE_boolean("is_training_flag",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters") #256--->512
tf.app.flags.DEFINE_string("word2vec_model_path","word2vec-title-desc.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
filter_sizes=[6,7,8]

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #trainX, trainY, testX, testY = None, None, None, None
    #vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, _= create_vocabulary(FLAGS.traning_data_path,FLAGS.vocab_size,name_scope=FLAGS.name_scope)
    word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY=load_data(FLAGS.cache_file_h5py, FLAGS.cache_file_pickle)
    vocab_size = len(word2index);print("cnn_model.vocab_size:",vocab_size);num_classes=len(label2index);print("num_classes:",num_classes)
    num_examples,FLAGS.sentence_len=trainX.shape
    print("num_examples of training:",num_examples,";sentence_len:",FLAGS.sentence_len)
    #train, test= load_data_multilabel(FLAGS.traning_data_path,vocabulary_word2index, vocabulary_label2index,FLAGS.sentence_len)
    #trainX, trainY = train;testX, testY = test
    #print some message for debug purpose
    print("trainX[0:10]:", trainX[0:10])
    print("trainY[0]:", trainY[0:10])
    train_y_short = get_target_label_short(trainY[0])
    print("train_y_short:", train_y_short)

    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        textCNN=TextCNN(filter_sizes,FLAGS.num_filters,num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,multi_label_flag=FLAGS.multi_label_flag)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            #for i in range(3): #decay learning rate if necessary.
            #    print(i,"Going to decay learning rate by half.")
            #    sess.run(textCNN.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                index2word={v:k for k,v in word2index.items()}
                assign_pretrained_word_embedding(sess, index2word, vocab_size, textCNN,FLAGS.word2vec_model_path)
        curr_epoch=sess.run(textCNN.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        iteration=0
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, counter =  0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                iteration=iteration+1
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])
                feed_dict = {textCNN.input_x: trainX[start:end],textCNN.dropout_keep_prob: 0.8,textCNN.is_training_flag:FLAGS.is_training_flag}
                if not FLAGS.multi_label_flag:
                    feed_dict[textCNN.input_y] = trainY[start:end]
                else:
                    feed_dict[textCNN.input_y_multilabel]=trainY[start:end]
                curr_loss,lr,_=sess.run([textCNN.loss_val,textCNN.learning_rate,textCNN.train_op],feed_dict)
                loss,counter=loss+curr_loss,counter+1
                if counter %50==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" %(epoch,counter,loss/float(counter),lr))

                ########################################################################################################
                if start%(3000*FLAGS.batch_size)==0: # eval every 3000 steps.
                    eval_loss, f1_score,f1_micro,f1_macro = do_eval(sess, textCNN, vaildX, vaildY,num_classes)
                    print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (epoch, eval_loss, f1_score,f1_micro,f1_macro))
                    # save model to checkpoint
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    print("Going to save model..")
                    saver.save(sess, save_path, global_step=epoch)
                ########################################################################################################
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(textCNN.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                eval_loss,f1_score,f1_micro,f1_macro=do_eval(sess,textCNN,testX,testY,num_classes)
                print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (epoch,eval_loss,f1_score,f1_micro,f1_macro))
                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss,f1_score,f1_micro,f1_macro = do_eval(sess, textCNN, testX, testY,num_classes)
        print("Test Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % ( test_loss,f1_score,f1_micro,f1_macro))
    pass


# 在验证集上做验证，报告损失、精确度
def do_eval(sess,textCNN,evalX,evalY,num_classes):
    evalX=evalX[0:3000]
    evalY=evalY[0:3000]
    number_examples=len(evalX)
    eval_loss,eval_counter,eval_f1_score,eval_p,eval_r=0.0,0,0.0,0.0,0.0
    batch_size=1
    label_dict_confuse_matrix=init_label_dict(num_classes)
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.input_y_multilabel:evalY[start:end],textCNN.dropout_keep_prob: 1.0,
                     textCNN.is_training_flag: False}
        curr_eval_loss, logits= sess.run([textCNN.loss_val,textCNN.logits],feed_dict)#curr_eval_acc--->textCNN.accuracy
        predict_y = get_label_using_logits(logits[0])
        target_y= get_target_label_short(evalY[start:end][0])
        #f1_score,p,r=compute_f1_score(list(label_list_top5), evalY[start:end][0])
        label_dict_confuse_matrix=compute_confuse_matrix(target_y, predict_y, label_dict_confuse_matrix)
        eval_loss,eval_counter=eval_loss+curr_eval_loss,eval_counter+1

    f1_micro,f1_macro=compute_micro_macro(label_dict_confuse_matrix) #label_dict_accusation is a dict, key is: accusation,value is: (TP,FP,FN). where TP is number of True Positive
    f1_score=(f1_micro+f1_macro)/2.0
    return eval_loss/float(eval_counter),f1_score,f1_micro,f1_macro

#######################################
def compute_f1_score(predict_y,eval_y):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    f1_score=0.0
    p_5=0.0
    r_5=0.0
    return f1_score,p_5,r_5

def compute_f1_score_removed(label_list_top5,eval_y):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    num_correct_label=0
    eval_y_short=get_target_label_short(eval_y)
    for label_predict in label_list_top5:
        if label_predict in eval_y_short:
            num_correct_label=num_correct_label+1
    #P@5=Precision@5
    num_labels_predicted=len(label_list_top5)
    all_real_labels=len(eval_y_short)
    p_5=num_correct_label/num_labels_predicted
    #R@5=Recall@5
    r_5=num_correct_label/all_real_labels
    f1_score=2.0*p_5*r_5/(p_5+r_5+0.000001)
    return f1_score,p_5,r_5

random_number=1000
def compute_confuse_matrix(target_y,predict_y,label_dict,name='default'):
    """
    compute true postive(TP), false postive(FP), false negative(FN) given target lable and predict label
    :param target_y:
    :param predict_y:
    :param label_dict {label:(TP,FP,FN)}
    :return: macro_f1(a scalar),micro_f1(a scalar)
    """
    #1.get target label and predict label
    if random.choice([x for x in range(random_number)]) ==1:
        print(name+".target_y:",target_y,";predict_y:",predict_y) #debug purpose

    #2.count number of TP,FP,FN for each class
    y_labels_unique=[]
    y_labels_unique.extend(target_y)
    y_labels_unique.extend(predict_y)
    y_labels_unique=list(set(y_labels_unique))
    for i,label in enumerate(y_labels_unique): #e.g. label=2
        TP, FP, FN = label_dict[label]
        if label in predict_y and label in target_y:#predict=1,truth=1 (TP)
            TP=TP+1
        elif label in predict_y and label not in target_y:#predict=1,truth=0(FP)
            FP=FP+1
        elif label not in predict_y and label in target_y:#predict=0,truth=1(FN)
            FN=FN+1
        label_dict[label] = (TP, FP, FN)
    return label_dict

def compute_micro_macro(label_dict):
    """
    compute f1 of micro and macro
    :param label_dict:
    :return: f1_micro,f1_macro: scalar, scalar
    """
    f1_micro = compute_f1_micro_use_TFFPFN(label_dict)
    f1_macro= compute_f1_macro_use_TFFPFN(label_dict)
    return f1_micro,f1_macro

def compute_TF_FP_FN_micro(label_dict):
    """
    compute micro FP,FP,FN
    :param label_dict_accusation: a dict. {label:(TP, FP, FN)}
    :return:TP_micro,FP_micro,FN_micro
    """
    TP_micro,FP_micro,FN_micro=0.0,0.0,0.0
    for label,tuplee in label_dict.items():
        TP,FP,FN=tuplee
        TP_micro=TP_micro+TP
        FP_micro=FP_micro+FP
        FN_micro=FN_micro+FN
    return TP_micro,FP_micro,FN_micro
def compute_f1_micro_use_TFFPFN(label_dict):
    """
    compute f1_micro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_micro: a scalar
    """
    TF_micro_accusation, FP_micro_accusation, FN_micro_accusation =compute_TF_FP_FN_micro(label_dict)
    f1_micro_accusation = compute_f1(TF_micro_accusation, FP_micro_accusation, FN_micro_accusation,'micro')
    return f1_micro_accusation

def compute_f1_macro_use_TFFPFN(label_dict):
    """
    compute f1_macro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_macro
    """
    f1_dict= {}
    num_classes=len(label_dict)
    for label, tuplee in label_dict.items():
        TP,FP,FN=tuplee
        f1_score_onelabel=compute_f1(TP,FP,FN,'macro')
        f1_dict[label]=f1_score_onelabel
    f1_score_sum=0.0
    for label,f1_score in f1_dict.items():
        f1_score_sum=f1_score_sum+f1_score
    f1_score=f1_score_sum/float(num_classes)
    return f1_score

small_value=0.00001
def compute_f1(TP,FP,FN,compute_type):
    """
    compute f1
    :param TP_micro: number.e.g. 200
    :param FP_micro: number.e.g. 200
    :param FN_micro: number.e.g. 200
    :return: f1_score: a scalar
    """
    precison=TP/(TP+FP+small_value)
    recall=TP/(TP+FN+small_value)
    f1_score=(2*precison*recall)/(precison+recall+small_value)

    if random.choice([x for x in range(500)]) == 1:print(compute_type,"precison:",str(precison),";recall:",str(recall),";f1_score:",f1_score)

    return f1_score
def init_label_dict(num_classes):
    """
    init label dict. this dict will be used to save TP,FP,FN
    :param num_classes:
    :return: label_dict: a dict. {label_index:(0,0,0)}
    """
    label_dict={}
    for i in range(num_classes):
        label_dict[i]=(0,0,0)
    return label_dict

def get_target_label_short(eval_y):
    eval_y_short=[] #will be like:[22,642,1391]
    for index,label in enumerate(eval_y):
        if label>0:
            eval_y_short.append(index)
    return eval_y_short

#get top5 predicted labels
def get_label_using_logits(logits,top_number=5):
    # index_list=np.argsort(logits)[-top_number:]
    #vindex_list=index_list[::-1]
    y_predict_labels = [i for i in range(len(logits)) if logits[i] >= 0.50]  # TODO 0.5PW e.g.[2,12,13,10]
    if len(y_predict_labels) < 1: y_predict_labels = [np.argmax(logits)]

    return y_predict_labels

#统计预测的准确率
def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

##################################################

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textCNN,word2vec_model_path):
    import word2vec # we put import here so that many people who do not use word2vec do not need to install this package. you can move import to the beginning of this file.
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(2, vocab_size):  # loop each word. notice that the first two words are pad and unknown token
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

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
    #print(train_X)
    #f_data.close()

    word2index, label2index=None,None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index=pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    return word2index, label2index,train_X,train_Y,vaild_X,valid_Y,test_X,test_Y
if __name__ == "__main__":
    tf.app.run()