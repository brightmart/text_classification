# -*- coding: utf-8 -*-
import random
def read_write(source_file_path,target_file_path):
    # 1.read file
    source_file_object=open(source_file_path,mode='r')
    target_file_object=open(target_file_path,mode='a')

    lines=source_file_object.readlines()
    random.shuffle(lines)
    # 2.write file
    for i,line in enumerate(lines):
        # line=w29368 w6 w7851 w24869 w35610 w111  __label__-5732564720446782766 2787171473654490487
        raw_list = line.strip().split("__label__")
        input_list = raw_list[0].strip().split(" ")
        input_list = [x.strip().replace(" ", "") for x in input_list if x != '']
        label_list = raw_list[1:]
        label_list=label_list[0].split(" ")
        label_list=[l.strip().replace(" ", "") for l in label_list if l != '']
        strings=" ".join(input_list)+' __label__'+" __label__".join(label_list)+"\n"
        target_file_object.write(strings)
        if i%10000==0:
           print(i,strings)
    target_file_object.close()
    source_file_object.close()
#source_file_path='/Users/xuliang/Downloads/train-zhihu6-title-desc-folder.txt/train-zhihu6-title-desc.txt'
#target_file_path='/Users/xuliang/Downloads/train-zhihu6-title-desc-folder.txt/train-zhihu-title-desc-multiple-label-v6.txt'

source_file_path='/Users/xuliang/Downloads/train-zhihu6-title-desc-folder.txt/test-zhihu6-title-desc.txt'
target_file_path='/Users/xuliang/Downloads/train-zhihu6-title-desc-folder.txt/test-zhihu-title-desc-multiple-label-v6.txt'
read_write(source_file_path,target_file_path)
