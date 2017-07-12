# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

#准备预测需要的数据.每一行作为问题的表示,写到文件中.
#prepreing prediction data. structure is just the same as training data.
#proces question. for every question form a a list of string to reprensent it.
import codecs
print("proces question started. get question representation...")
target_filename='test-muying-forpredict-v4only-title.txt'
target_file_predict = codecs.open(target_filename, 'a', 'utf8')
qv='question_eval_set.txt'
q_filev = codecs.open(qv, 'r', 'utf8')
q_linesv=q_filev.readlines()
questionid_words_representationv={}
question_representationv=[]
question_representationv_list=[]
for i,line in enumerate(q_linesv):
    element_lists=line.split('\t') #['c324,c39','w305...','c']
    question_id=element_lists[0]
    question_representationv=[x for x in element_lists[2].split(",")] # TODO +[x for x in element_lists[1].split(",")]
    #print("question_representation:",question_representationv)
    questionid_words_representationv[question_id]=question_representationv
    question_representationv_list.append(question_representationv)
    #if i>5:
    #    break
    question_representationv_=" ".join(question_representationv)
    target_file_predict.write(question_representationv_+"\n")
target_file_predict.close()
print("proces question ended...")