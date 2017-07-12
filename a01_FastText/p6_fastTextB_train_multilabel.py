# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from p6_fastTextB_model_multilabel import fastTextB as fastText
from p4_zhihu_load_data import load_data_with_multilabels,create_voabulary,create_voabulary_label
from tflearn.data_utils import to_categorical, pad_sequences
import os
import word2vec

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("label_size",1999,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.") #512批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 20000, "how many steps before decay learning rate.") #批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少
tf.app.flags.DEFINE_integer("num_sampled",10,"number of noise sampling") #100
tf.app.flags.DEFINE_string("ckpt_dir","fast_text_checkpoint_multi/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",300,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size") #100
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",16,"embedding size")
tf.app.flags.DEFINE_integer("validate_every", 3, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_string("training_path", '/home/xul/xul/9_fastTextB/training-data/test-zhihu6-only-title-multilabel-trigram.txt', "location of traning data.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    trainX, trainY, testX, testY = None, None, None, None
    vocabulary_word2index, vocabulary_index2word = create_voabulary()
    vocab_size = len(vocabulary_word2index)
    vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label()
    train,test = load_data_with_multilabels(vocabulary_word2index, vocabulary_word2index_label,FLAGS.training_path) #[1,11,3,1998,1998]
    trainX, trainY= train #TODO trainY1999
    testX, testY = test #TODO testY1999
    print("testX.shape:", np.array(testX).shape);print("testY.shape:", np.array(testY).shape)  # 2500个label
    # 2.Data preprocessing
    # Sequence padding
    print("start padding & transform to one hot...")
    trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    print("end padding & transform to one hot...")

    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        fast_text=fastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,FLAGS.num_sampled,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, fast_text)

        curr_epoch=sess.run(fast_text.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        for epoch in range(curr_epoch,FLAGS.num_epochs):#range(start,stop,step_size)
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end]) #2d-array. each element slength is a 100.
                    print("trainY[start:end]:",trainY[start:end]) #a list,each element is a list.element:may be has 1,2,3,4,5 labels.
                    #print("trainY1999[start:end]:",trainY1999[start:end])
                curr_loss,_=sess.run([fast_text.loss_val,fast_text.train_op],feed_dict={fast_text.sentence:trainX[start:end],fast_text.labels:trainY[start:end],}) #fast_text.labels_l1999:trainY1999[start:end]
                loss,counter=loss+curr_loss,counter+1 #acc+curr_acc,
                if counter %500==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f" %(epoch,counter,loss/float(counter))) #\tTrain Accuracy:%.3f--->,acc/float(counter)

            #epoch increment
            print("going to increment epoch counter....")
            sess.run(fast_text.epoch_increment)

            # 4.validation
            print("epoch:",epoch,"validate_every:",FLAGS.validate_every,"validate or not:",(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                eval_loss,eval_accuracy=do_eval(sess,fast_text,testX,testY,batch_size,vocabulary_index2word_label) #testY1999,eval_acc
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch,eval_loss,eval_accuracy)) #,\tValidation Accuracy: %.3f--->eval_acc
                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch) #fast_text.epoch_step

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, test_acc = do_eval(sess, fast_text, testX, testY,batch_size,vocabulary_index2word_label) #testY1999
    pass

# 在验证集上做验证，报告损失、精确度
def do_eval(sess,fast_text,evalX,evalY,batch_size,vocabulary_index2word_label): #evalY1999
    number_examples=len(evalX)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    batch_size=1
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        curr_eval_loss,logits_ = sess.run([fast_text.loss_val,fast_text.logits], #curr_eval_acc-->fast_text.accuracy
                                          feed_dict={fast_text.sentence: evalX[start:end],fast_text.labels: evalY[start:end]}) #,fast_text.labels_l1999:evalY1999[start:end]
        print("do_eval.logits_",logits_)
        label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
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

if __name__ == "__main__":
    tf.app.run()