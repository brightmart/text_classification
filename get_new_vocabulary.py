import os

filter_list=['地址','目的地','出发地','消息内容','起始地点','偏移量','方向']
def get_new_vocab():
    dict_result={}
    #1.get files under the data path ".txt"
    files_name_list = os.listdir('.')
    for f_name in files_name_list:
        ignore_flag=False
        for filter_name in filter_list:
            if filter_name in f_name:
                ignore_flag=True
        if ignore_flag:#2.ignore those files contain key words the filter_list
            print("going to ignore:",f_name)
            continue
        #3.read data
        file_size=os.path.getsize(f_name)/1000.0
        if file_size<=71:
            #read all content to dict
            pass
        else:
            #read only last 7000
        #4.ignore content which name is longer than 5.
        #5.put key,value in dict:dict[value]=key

    return dict_result

def put_last7000_to_dict(lines,dict_result):
    lines_reversed=lines.reverse()

def save_dict_to_file_system(dict_words,f):
    for key,value in dict_words:
        f.write(value+"\n")

