from  p7_TextCNN_predict import get_logits_with_value_by_input
from p7_TextCNN_predict_exp import get_logits_with_value_by_input_exp
import tensorflow as tf
def main(_):
    for start in range(217360):
        end=start+1
        label_list,p_list=get_logits_with_value_by_input(start,end)
        label_list_exp, p_list_exp=get_logits_with_value_by_input_exp(start,end)

        if start<5:
            print("----------------------------------------------------")
            print(start,"label_list0:",label_list,"p_list0:",p_list)
            print(start,"label_list1:", label_list_exp, "p_list1:", p_list_exp)
        else:
            break



if __name__ == "__main__":
    tf.app.run()