# -*- coding: utf-8 -*-
"""
training the model.
process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
fast text. using: very simple model;n-gram to captrue location information;h-softmax to speed up training/inference
for the n-gram,you can use data_util to generate. see method process_one_sentence_to_get_ui_bi_tri_gram under aa1_data_util/data_util_zhihu.py

"""
import tensorflow as tf
import numpy as np
from p6_fastTextB_model_multilabel import fastTextB as fastText
#from p4_zhihu_load_data import load_data_with_multilabels,create_voabulary,create_voabulary_label
#from tflearn.data_utils import to_categorical, pad_sequences
import os
import word2vec
import pickle
import h5py
#configuration
FLAGS=tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer("label_size",1999,"number of label")
tf.app.flags.DEFINE_string("cache_file_h5py","../data/ieee_zhihu_cup/data.h5","path of training/validation/test data.") #../data/sample_multiple_label.txt
tf.app.flags.DEFINE_string("cache_file_pickle","../data/ieee_zhihu_cup/vocab_label.pik","path of vocabulary and label files") #../data/sample_multiple_label.txt

tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.") #512批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 20000, "how many steps before decay learning rate.") #批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少
tf.app.flags.DEFINE_integer("num_sampled",10,"number of noise sampling") #100
tf.app.flags.DEFINE_string("ckpt_dir","fast_text_checkpoint_multi/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",200,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size") #100
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",25,"embedding size")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
#tf.app.flags.DEFINE_string("training_path", '/home/xul/xul/9_fastTextB/training-data/test-zhihu6-only-title-multilabel-trigram.txt', "location of traning data.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    trainX, trainY, testX, testY = None, None, None, None
    #vocabulary_word2index, vocabulary_index2word = create_voabulary()
    #vocab_size = len(vocabulary_word2index)
    #vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label()
    #train,test = load_data_with_multilabels(vocabulary_word2index, vocabulary_word2index_label,FLAGS.training_path) #[1,11,3,1998,1998]
    #trainX, trainY= train #TODO trainY1999
    #testX, testY = test #TODO testY1999
    #print("testX.shape:", np.array(testX).shape);print("testY.shape:", np.array(testY).shape)  # 2500个label
    # 2.Data preprocessing
    # Sequence padding
    #print("start padding & transform to one hot...")
    #trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    #testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    #print("end padding & transform to one hot...")
    word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY=load_data(FLAGS.cache_file_h5py, FLAGS.cache_file_pickle)
    index2label={v:k for k,v in label2index.items()}
    vocab_size = len(word2index);print("cnn_model.vocab_size:",vocab_size);num_classes=len(label2index);print("num_classes:",num_classes)
    num_examples,FLAGS.sentence_len=trainX.shape
    print("num_examples of training:",num_examples,";sentence_len:",FLAGS.sentence_len)

    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        fast_text=fastText(num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,FLAGS.num_sampled,FLAGS.sentence_len,
                           vocab_size,FLAGS.embed_size,FLAGS.is_training)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                vocabulary_index2word={v:k for k,v in word2index.items()}
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, fast_text)

        curr_epoch=sess.run(fast_text.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        for epoch in range(curr_epoch,FLAGS.num_epochs):#range(start,stop,step_size)
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                #train_Y_batch=process_labels(trainY[start:end],number=start)
                curr_loss,current_l2_loss,_=sess.run([fast_text.loss_val,fast_text.l2_losses,fast_text.train_op],
                                                     feed_dict={fast_text.sentence:trainX[start:end],fast_text.labels_l1999:trainY[start:end]}) #fast_text.labels_l1999:trainY1999[start:end]
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end]) #2d-array. each element slength is a 100.
                    print("train_Y_batch:",trainY[start:end]) #a list,each element is a list.element:may be has 1,2,3,4,5 labels.
                    #print("trainY1999[start:end]:",trainY1999[start:end])
                loss,counter=loss+curr_loss,counter+1 #acc+curr_acc,
                if counter %50==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tL2 Loss:%.3f" %(epoch,counter,loss/float(counter),current_l2_loss)) #\tTrain Accuracy:%.3f--->,acc/float(counter)

                if start%(1000*FLAGS.batch_size)==0:
                    eval_loss, eval_accuracy = do_eval(sess, fast_text, vaildX, vaildY, batch_size,index2label)  # testY1999,eval_acc
                    print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_accuracy))  # ,\tValidation Accuracy: %.3f--->eval_acc
                    # save model to checkpoint
                    if start%(6000*FLAGS.batch_size)==0:
                        print("Going to save checkpoint.")
                        save_path = FLAGS.ckpt_dir + "model.ckpt"
                        saver.save(sess, save_path, global_step=epoch)  # fast_text.epoch_step
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(fast_text.epoch_increment)

            # 4.validation
            print("epoch:",epoch,"validate_every:",FLAGS.validate_every,"validate or not:",(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                eval_loss,eval_accuracy=do_eval(sess,fast_text,vaildX,vaildY,batch_size,index2label) #testY1999,eval_acc
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch,eval_loss,eval_accuracy)) #,\tValidation Accuracy: %.3f--->eval_acc
                #save model to checkpoint
                print("Going to save checkpoint.")
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch) #fast_text.epoch_step

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, test_acc = do_eval(sess, fast_text, testX, testY,batch_size,index2label) #testY1999
    pass

# 在验证集上做验证，报告损失、精确度
def do_eval(sess,fast_text,evalX,evalY,batch_size,vocabulary_index2word_label): #evalY1999
    evalX=evalX[0:3000]
    evalY=evalY[0:3000]
    number_examples,labels=evalX.shape
    print("number_examples for validation:",number_examples)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    batch_size=1
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        evalY_batch=process_labels(evalY[start:end])
        curr_eval_loss,logit = sess.run([fast_text.loss_val,fast_text.logits], #curr_eval_acc-->fast_text.accuracy
                                          feed_dict={fast_text.sentence: evalX[start:end],fast_text.labels_l1999: evalY[start:end]}) #,fast_text.labels_l1999:evalY1999[start:end]
        #print("do_eval.logits_",logits_.shape)
        label_list_top5 = get_label_using_logits(logit[0], vocabulary_index2word_label)
        curr_eval_acc=calculate_accuracy(list(label_list_top5),evalY_batch[0] ,eval_counter) # evalY[start:end][0]
        eval_loss,eval_counter,eval_acc=eval_loss+curr_eval_loss,eval_counter+1,eval_acc+curr_eval_acc

    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,fast_text):
    print("using pre-trained word emebedding.started...")
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    word2vec_model = word2vec.load('zhihu-word2vec-multilabel.bin-100', kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
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
    t_assign_embedding = tf.assign(fast_text.Embedding,
                                   word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

#从logits中取出前五 get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=5):
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    #label_list=[]
    #for index in index_list:
    #    label=vocabulary_index2word_label[index]
    #    label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return index_list

#统计预测的准确率
def calculate_accuracy(labels_predicted, labels,eval_counter):
    if eval_counter<10:
        print("labels_predicted:",labels_predicted," ;labels:",labels)
    count = 0
    label_dict = {x: x for x in labels}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

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


    word2index, label2index=None,None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index=pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    return word2index, label2index,train_X,train_Y,vaild_X,valid_Y,test_X,test_Y

def process_labels(trainY_batch,require_size=5,number=None):
    """
    process labels to get fixed size labels given a spense label
    :param trainY_batch:
    :return:
    """
    #print("###trainY_batch:",trainY_batch)
    num_examples,_=trainY_batch.shape
    trainY_batch_result=np.zeros((num_examples,require_size),dtype=int)

    for index in range(num_examples):
        y_list_sparse=trainY_batch[index]
        y_list_dense = [i for i, label in enumerate(y_list_sparse) if int(label) == 1]
        y_list=proces_label_to_algin(y_list_dense,require_size=require_size)
        trainY_batch_result[index]=y_list
        if number is not None and number%30==0:
            pass
            #print("####0.y_list_sparse:",y_list_sparse)
            #print("####1.y_list_dense:",y_list_dense)
            #print("####2.y_list:",y_list) # 1.label_index: [315] ;2.y_list: [315, 315, 315, 315, 315] ;3.y_list: [0. 0. 0. ... 0. 0. 0.]
    if number is not None and number % 30 == 0:
        #print("###3trainY_batch_result:",trainY_batch_result)
        pass
    return trainY_batch_result

def proces_label_to_algin(ys_list,require_size=5):
    """
    given a list of labels, process it to fixed size('require_size')
    :param ys_list: a list
    :return: a list
    """
    ys_list_result=[0 for x in range(require_size)]
    if len(ys_list)>=require_size: #超长
        ys_list_result=ys_list[0:require_size]
    else:#太短
       if len(ys_list)==1:
           ys_list_result =[ys_list[0] for x in range(require_size)]
       elif len(ys_list)==2:
           ys_list_result = [ys_list[0],ys_list[0],ys_list[0],ys_list[1],ys_list[1]]
       elif len(ys_list) == 3:
           ys_list_result = [ys_list[0], ys_list[0], ys_list[1], ys_list[1], ys_list[2]]
       elif len(ys_list) == 4:
           ys_list_result = [ys_list[0], ys_list[0], ys_list[1], ys_list[2], ys_list[3]]
    return ys_list_result
if __name__ == "__main__":
    tf.app.run()