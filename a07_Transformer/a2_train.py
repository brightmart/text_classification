# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from a07_Transformer import  Transformer
from data_util_zhihu import load_data_multilabel_new,create_voabulary,create_voabulary_label
from tflearn.data_utils import to_categorical, pad_sequences
import os,math
import word2vec
import pickle

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",1999+3,"number of label") #3 ADDITIONAL TOKEN: _GO,_END,_PAD
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.") #批处理的大小 32-->128-->512
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.87一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint_transformer/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",25,"max sentence length") #25
tf.app.flags.DEFINE_integer("embed_size",512,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_integer("validate_step", 1000, "how many step to validate.") #1500做一次检验
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
#tf.app.flags.DEFINE_string("cache_path","text_cnn_checkpoint/data_cache.pik","checkpoint location for the model")
#train-zhihu4-only-title-all.txt
tf.app.flags.DEFINE_string("traning_data_path","train-zhihu4-only-title-all.txt","path of traning data.") #O.K.train-zhihu4-only-title-all.txt-->training-data/test-zhihu4-only-title.txt--->'training-data/train-zhihu5-only-title-multilabel.txt'
tf.app.flags.DEFINE_string("word2vec_model_path","zhihu-word2vec-title-desc.bin-512","word2vec's vocabulary and vectors") #zhihu-word2vec.bin-100-->zhihu-word2vec-multilabel-minicount15.bin-100
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.") #set this false. becase we are using it is a sequence of token here.
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")

tf.app.flags.DEFINE_integer("d_model",512,"hidden size")
tf.app.flags.DEFINE_integer("d_k",64,"hidden size")
tf.app.flags.DEFINE_integer("d_v",64,"hidden size")
tf.app.flags.DEFINE_integer("h",8,"hidden size")
tf.app.flags.DEFINE_integer("num_layer",1,"hidden size") #6
tf.app.flags.DEFINE_integer("decoder_sent_length",25,"length of decoder inputs") #decoder sentence length should be 6 here.

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #1.load data(X:list of lint,y:int).
    #if os.path.exists(FLAGS.cache_path):  # 如果文件系统中存在，那么加载故事（词汇表索引化的）
    #    with open(FLAGS.cache_path, 'r') as data_f:
    #        trainX, trainY, testX, testY, vocabulary_index2word=pickle.load(data_f)
    #        vocab_size=len(vocabulary_index2word)
    #else:
    if 1==1:
        trainX, trainY, testX, testY = None, None, None, None
        vocabulary_word2index, vocabulary_index2word = create_voabulary(word2vec_model_path=FLAGS.word2vec_model_path,name_scope="transformer") #simple='simple'
        vocab_size = len(vocabulary_word2index)
        print("transformer.vocab_size:",vocab_size)
        vocabulary_word2index_label,vocabulary_index2word_label = create_voabulary_label(name_scope="transformer",use_seq2seq=True)
        if FLAGS.multi_label_flag:
            FLAGS.traning_data_path='training-data/train-zhihu6-title-desc.txt' #train
        train,test,_=load_data_multilabel_new(vocabulary_word2index,vocabulary_word2index_label,multi_label_flag=FLAGS.multi_label_flag,
                                              use_seq2seq=True,traning_data_path=FLAGS.traning_data_path,seq2seq_label_length=FLAGS.decoder_sent_length) #TODO
        trainX, trainY,train_decoder_input = train
        testX, testY,test_decoder_input = test

        print("trainY:",trainY[0:10])
        print("train_decoder_input:",train_decoder_input[0:10])
        # 2.Data preprocessing.Sequence padding
        print("start padding & transform to one hot...")
        trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
        testX = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
        #with open(FLAGS.cache_path, 'w') as data_f: #save data to cache file, so we can use it next time quickly.
        #    pickle.dump((trainX,trainY,testX,testY,vocabulary_index2word),data_f)
        print("trainX[0]:", trainX[0]) #;print("trainY[0]:", trainY[0])
        # Converting labels to binary vectors
        print("end padding & transform to one hot...")
    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        model=Transformer(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length,
                 vocab_size, FLAGS.embed_size,FLAGS.d_model,FLAGS.d_k,FLAGS.d_v,FLAGS.h,FLAGS.num_layer,FLAGS.is_training,
                          decoder_sent_length=FLAGS.sequence_length,l2_lambda=FLAGS.l2_lambda) #TODO decoder_sent_length=FLAGS.sequence_length
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, model,word2vec_model_path=FLAGS.word2vec_model_path)
        curr_epoch=sess.run(model.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        print("number_of_training_data:",number_of_training_data)
        previous_eval_loss=10000
        best_eval_loss=10000
        batch_size=FLAGS.batch_size
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])#;print("trainY[start:end]:",trainY[start:end])
                feed_dict = {model.input_x: trainX[start:end],model.dropout_keep_prob: 0.5}
                if not FLAGS.multi_label_flag:
                    feed_dict[model.input_y] = trainY[start:end]
                else:
                    feed_dict[model.input_y_label]=trainY[start:end]
                    feed_dict[model.decoder_input] = train_decoder_input[start:end]
                curr_loss,curr_acc,_=sess.run([model.loss_val,model.accuracy,model.train_op],feed_dict) #curr_acc--->TextCNN.accuracy
                loss,counter,acc=loss+curr_loss,counter+1,acc+curr_acc
                if counter %50==0:
                    print("transformer==>Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch,counter,math.exp(loss/float(counter)) if (loss/float(counter))<20 else 10000.000,acc/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)
                ##VALIDATION VALIDATION VALIDATION PART######################################################################################################
                if FLAGS.batch_size!=0 and (start%(FLAGS.validate_step*FLAGS.batch_size)==0): #(epoch % FLAGS.validate_every) or  if epoch % FLAGS.validate_every == 0:
                    eval_loss, eval_acc = do_eval(sess, model, testX, testY, batch_size,vocabulary_index2word_label,eval_decoder_input=test_decoder_input)
                    print("transformer.validation.part. previous_eval_loss:", math.exp(previous_eval_loss) if previous_eval_loss<20 else 10000.000,";current_eval_loss:", math.exp(eval_loss) if eval_loss<20 else 10000.000)
                    if eval_loss > previous_eval_loss: #if loss is not decreasing
                        # reduce the learning rate by a factor of 0.5
                        print("transformer==>validation.part.going to reduce the learning rate.")
                        learning_rate1 = sess.run(model.learning_rate)
                        lrr=sess.run([model.learning_rate_decay_half_op])
                        learning_rate2 = sess.run(model.learning_rate)
                        print("transformer==>validation.part.learning_rate1:", learning_rate1, " ;learning_rate2:",learning_rate2)
                    #print("HierAtten==>Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                    else:# loss is decreasing
                        if eval_loss<best_eval_loss:
                            print("transformer==>going to save the model.eval_loss:",math.exp(eval_loss) if eval_loss<20 else 10000.000,";best_eval_loss:",math.exp(best_eval_loss) if best_eval_loss<20 else 10000.000)
                            # save model to checkpoint
                            save_path = FLAGS.ckpt_dir + "model.ckpt"
                            saver.save(sess, save_path, global_step=epoch)
                            best_eval_loss=eval_loss
                    previous_eval_loss = eval_loss
                ##VALIDATION VALIDATION VALIDATION PART######################################################################################################

            #epoch increment
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, test_acc = do_eval(sess, model, testX, testY, batch_size,vocabulary_index2word_label,eval_decoder_input=test_decoder_input)
    pass

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,model,word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
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
    t_assign_embedding = tf.assign(model.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

# 在验证集上做验证，报告损失、精确度
def do_eval(sess,model,evalX,evalY,batch_size,vocabulary_index2word_label,eval_decoder_input=None):
    #ii=0
    number_examples=len(evalX)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {model.input_x: evalX[start:end], model.dropout_keep_prob: 1.0}
        if not FLAGS.multi_label_flag:
            feed_dict[model.input_y] = evalY[start:end]
        else:
            feed_dict[model.input_y_label] = evalY[start:end]
            feed_dict[model.decoder_input] = eval_decoder_input[start:end]
        curr_eval_loss, logits,curr_eval_acc,pred= sess.run([model.loss_val,model.logits,model.accuracy,model.predictions],feed_dict)#curr_eval_acc--->textCNN.accuracy
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
        #if ii<20:
            #print("1.evalX[start:end]:",evalX[start:end])
            #print("2.evalY[start:end]:", evalY[start:end])
            #print("3.pred:",pred)
            #ii=ii+1
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

if __name__ == "__main__":
    tf.app.run()