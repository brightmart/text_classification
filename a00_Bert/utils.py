# -*- coding: utf-8 -*-

import pickle
import h5py
import os
import numpy as np
import random

random_number=300

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

def get_target_label_short_batch(eval_y_big): # tested.
    eval_y_short_big=[] #will be like:[22,642,1391]
    for ind, eval_y in enumerate(eval_y_big):
        eval_y_short=[]
        for index,label in enumerate(eval_y):
            if label>0:
                eval_y_short.append(index)
        eval_y_short_big.append(eval_y_short)
    return eval_y_short_big

#eval_y_big=np.zeros((3,6))
#eval_y_big[0,5]=1
#eval_y_big[0,0]=1
#eval_y_big[1,0]=1
#eval_y_big[1,1]=1
#print("eval_y_big:",eval_y_big)
#result=get_target_label_short_batch(eval_y_big)
#print("result:",result)

#get top5 predicted labels
def get_label_using_prob(prob,top_number=5):
    y_predict_labels = [i for i in range(len(prob)) if prob[i] >= 0.50]  # TODO 0.5PW e.g.[2,12,13,10]
    if len(y_predict_labels) < 1:
        y_predict_labels = [np.argmax(prob)]
    return y_predict_labels

def get_label_using_logits_batch(prob,top_number=5): # tested.
    result_labels=[]
    for i in range(len(prob)):
        single_prob=prob[i]
        labels=get_label_using_prob(single_prob)
        result_labels.append(labels)
    return result_labels

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

def compute_confuse_matrix_batch(y_targetlabel_list,y_logits_array,label_dict,name='default'):
    """
    compute confuse matrix for a batch
    :param y_targetlabel_list: a list; each element is a mulit-hot,e.g. [1,0,0,1,...]
    :param y_logits_array: a 2-d array. [batch_size,num_class]
    :param label_dict:{label:(TP, FP, FN)}
    :param name: a string for debug purpose
    :return:label_dict:{label:(TP, FP, FN)}
    """
    for i,y_targetlabel_list_single in enumerate(y_targetlabel_list):
        label_dict=compute_confuse_matrix(y_targetlabel_list_single,y_logits_array[i],label_dict,name=name)
    return label_dict