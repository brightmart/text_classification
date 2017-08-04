Text Classification
-------------------------------------------------------------------------
the purpose of this repository is to explore text classification methods in NLP with deep learning. 

it has all kinds of baseline models for text classificaiton.

it also support for multi-label classification where multi label associate with an sentence or document.

although many of these models are simple, and may not get you to top level of the task.but some of these models are very classic, so they may be good to serve as baseline models.

each model has a test function under model class. you can run it to performance toy task first. the model is indenpendent from dataset.

serveral modes here can also be used for modelling question answering (with or without context), or to do sequences generating. 

we explore two seq2seq model(seq2seq with attention,transformer-attention is all you need) to do text classification. and these two models can also be used for sequences generating and other tasks. if you task is a multi-label classification, you can cast the problem to sequences generating.

we implement two memory network. one is dynamic memory network. previously it reached state of art in question answering, sentiment analysis and sequence generating tasks. it is so called one model to do serveral different tasks, and reach high performance. it has four modules. the key component is episodic memory module. it use gate mechanism to performance attention, and use gated-gru to update episode memory, then it has another gru( in a vertical direction) to pefromance hidden state update. it has ability to do transitive inference.

the second memory network we implemented is recurrent entity network: tracking state of the world. it has blocks of key-value pairs as memory, run in parallel, which achieve new state of art. it can be used for modelling question answering with contexts(or history). for example, you can let the model to read some sentences(as context), and ask a question(as query), then ask the model to predict an answer; if you feed story same as query, then it can do classification task. 

if you need some sample data, you can find it in closed issues.

if you want to know more detail about dataset of text classification or task these models can be used, one of choose is below:
https://biendata.com/competition/zhihu/


Models:
-------------------------------------------------------------------------

1) fastText
2) TextCNN   
3) TextRNN    
4) RCNN     
5) Hierarchical Attention Network    
6) seq2seq with attention   
7) Transformer("Attend Is All You Need")
8) Dynamic Memory Network
9) EntityNetwork:tracking state of the world
10) Ensemble models

and other models:

1) BiLstmTextRelation;

2) twoCNNTextRelation;

3) BiLstmTextRelationTwoRNN

Performance
-------------------------------------------------------------------------

(mulit-label label prediction task,ask to prediction top5, 3 million training data,full score:0.5)

Model   | fastText|TextCNN|TextRNN| RCNN | HierAtteNet|Seq2seqAttn|EntityNet|DynamicMemory|Transformer
---     | ---     | ---   | ---   |---   |---         |---        |---      |---          |----
Score   | 0.362   |  0.405| 0.358 | 0.395| 0.398      |0.322      |0.400    |0.392        |0.322
Training| 10m     |  2h   |10h    | 2h   | 2h         |3h         |3h       |5h           |7h
--------------------------------------------------------------------------------------------------

 Ensemble of TextCNN,EntityNet,DynamicMemory: 0.411
 
 Ensemble EntityNet,DynamicMemory: 0.403
 
 Notice: 
 
 'm' stand for minutes; 'h' stand for hours;
 
'HierAtteNet' means Hierarchical Attention Networkk;

'Seq2seqAttn' means Seq2seq with attention;

'DynamicMemory' means DynamicMemoryNetwork;

'Transformer' stand for model from 'Attention Is All You Need'.

Useage:
-------------------------------------------------------------------------------------------------------
1) model is in xxx_model.py
2) run python xxx_train.py to train the model
3) run python xxx_predict.py to do inference(test).

Each model has a test method under the model class. you can run the test method first to check whether the model can work properly.

-------------------------------------------------------------------------

Environment:
-------------------------------------------------------------------------------------------------------
python 2.7+ tensorflow 1.1

(tensorflow 1.2 also works; most of models should also work fine in other tensorflow version, since we use very few features bond to certain version; if you use python 3.5, it will be fine as long as you change print/try catch function)

-------------------------------------------------------------------------

Notice:
-------------------------------------------------------------------------------------------------------
Some util function is in data_util.py; 
typical input like: "x1 x2 x3 x4 x5 __label__ 323434" where 'x1,x2' is words, '323434' is label;
it has a function to load and assign pretrained word embedding to the model,where word embedding is pretrained in word2vec or fastText. 

Models Detail:
-------------------------------------------------------------------------

1.fastText:  
-------------
implmentation of <a href="https://arxiv.org/abs/1607.01759">Bag of Tricks for Efficient Text Classification</a>
1) use bi-gram and/or tri-gram
2) use NCE loss to speed us softmax computation(not use hierarchy softmax as original paper)
result: performance is as good as paper, speed also very fast.

check: p5_fastTextB_model.py

-------------------------------------------------------------------------

2.TextCNN:
-------------
implementation of <a href="http://www.aclweb.org/anthology/D14-1181"> Convolutional Neural Networks for Sentence Classification </a>

Structure:embedding--->conv--->max pooling--->fully connected layer-------->softmax

check: p7_TextCNN_model.py

in order to get very good result with TextCNN, you also need to read carefully about this paper <a href="https://arxiv.org/abs/1510.03820">A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification</a>: it give you some insights of things that can affect performance. although you need to  change some settings according to your specific task.


-------------------------------------------------------------------------


3.TextRNN
-------------
Structure:embedding--->bi-directional lstm--->concat output--->average----->softmax

check: p8_TextRNN_model.py


-------------------------------------------------------------------------


4.BiLstmTextRelation
-------------
Structure same as TextRNN. but input is special designed. e.g.input:"how much is the computer? EOS price of laptop". where 'EOS' is a special
token spilted question1 and question2.

check:p9_BiLstmTextRelation_model.py


-------------------------------------------------------------------------


5.twoCNNTextRelation
-------------
Structure: first use two different convolutional to extract feature of two sentences. then concat two features. use linear
transform layer to out projection to target label, then softmax.

check: p9_twoCNNTextRelation_model.py


-------------------------------------------------------------------------


6.BiLstmTextRelationTwoRNN
-------------
Structure: one bi-directional lstm for one sentence(get output1), another bi-directional lstm for another sentence(get output2). then:
softmax(output1*M*output2)

check:p9_BiLstmTextRelationTwoRNN_model.py

for more detail you can go to: <a herf="http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow">Deep Learning for Chatbots, Part 2 – Implementing a Retrieval-Based Model in Tensorflow<a>


-------------------------------------------------------------------------


7.RCNN:
-------------
Recurrent convolutional neural network for text classification

implementation of <a href="https://scholar.google.com.hk/scholar?q=Recurrent+Convolutional+Neural+Networks+for+Text+Classification&hl=zh-CN&as_sdt=0&as_vis=1&oi=scholart&sa=X&ved=0ahUKEwjpx82cvqTUAhWHspQKHUbDBDYQgQMIITAA"> Recurrent Convolutional Neural Network for Text Classification </a>
 
structure:1)recurrent structure (convolutional layer) 2)max pooling 3) fully connected layer+softmax

it learn represenation of each word in the sentence or document with left side context and right side context:

representation current word=[left_side_context_vector,current_word_embedding,right_side_context_vecotor].

for left side context, it use a recurrent structure, a no-linearity transfrom of previous word and left side previous context; similarly to right side context.

check: p71_TextRCNN_model.py


-------------------------------------------------------------------------

8.Hierarchical Attention Network:
-------------
Implementation of <a href="https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf">Hierarchical Attention Networks for Document Classification</a>

Structure:

1)embedding 

2) Word Encoder: word level bi-directional GRU to get rich representation of words

3) Word Attention:word level attention to get important information in a sentence

4) Sentence Encoder: sentence level bi-directional GRU to get rich representation of sentences

5) Sentence Attetion: sentence level attention to get important sentence among sentences

5) FC+Softmax


Input of data: 

Generally speaking, input of this model should have serveral sentences instead of sinle sentence. shape is:[None,sentence_lenght]. where None means the batch_size.

In my training data, for each example, i have four parts. each part has same length. i concat four parts to form one single sentence. the model will split the sentence into four parts, to form a tensor with shape:[None,num_sentence,sentence_length]. where num_sentence is number of sentences(equal to 4, in my setting).

check:p1_HierarchicalAttention_model.py

-------------------------------------------------------------------------

9.Seq2seq with attention
-------------
Implementation seq2seq with attention derived from <a href="https://arxiv.org/pdf/1409.0473.pdf">NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE</a>

I.Structure:

1)embedding 2)bi-GRU too get rich representation from source sentences(forward & backward). 3)decoder with attention.

II.Input of data:

there are two kinds of three kinds of inputs:1)encoder inputs, which is a sentence; 2)decoder inputs, it is labels list with fixed length;3)target labels, it is also a list of labels.

for example, labels is:"L1 L2 L3 L4", then decoder inputs will be:[_GO,L1,L2,L2,L3,_PAD]; target label will be:[L1,L2,L3,L3,_END,_PAD]. length is fixed to 6, any exceed labels will be trancated, will pad if label is not enough to fill.

III.Attention Mechanism:

1) transfer encoder input list and hidden state of decoder

2) calculate similiarity of hidden state with each encoder input, to get possibility distribution for each encoder input.

3) weighted sum of encoder input based on possibility distribution.

   go though RNN Cell using this weight sum together with decoder input to get new hidden state

IV.How Vanilla Encoder Decoder Works:

the source sentence will be encoded using RNN as fixed size vector ("thought vector"). then during decoder:

1) when it is training, another RNN will be used to try to get a word by using this "thought vector"  as init state, and take input from decoder input at each timestamp. decoder start from special token "_GO". 
after one step is performanced, new hidden state will be get and together with new input, we can continue this process until we reach to a special token "_END". 
we can calculate loss by compute cross entropy loss of logits and target label. logits is get through a projection layer for the hidden state(for output of decoder step(in GRU we can just use hidden states from decoder as output).

2) when it is testing, there is no label. so we should feed the output we get from previous timestamp, and continue the process util we reached "_END" TOKEN.

V.Notices:

1) here i use two kinds of vocabularies. one is from words,used by encoder; another is for labels,used by decoder

2) for vocabulary of lables, i insert three special token:"_GO","_END","_PAD"; "_UNK" is not used, since all labels is pre-defined.

-------------------------------------------------------------------------

10.Transformer("Attention Is All You Need")
-------------
Status: it was able to do task classification. and able to generate reverse order of its sequences in toy task. you can check it by running test function in the model. check: a2_train_classification.py(train) or a2_transformer_classification.py(model)

we do it in parallell style.layer normalization,residual connection, and mask are also used in the model. 

For every building blocks, we include a test function in the each file below, and we've test each small piece successfully.

Sequence to sequence with attention is a typical model to solve sequence generation problem, such as translate, dialogue system. most of time, it use RNN as buidling block to do these tasks. util recently, people also apply convolutional Neural Network for sequence to sequence problem. Transformer, however, it perform these tasks solely on attention mechansim. it is fast and acheive new state-of-art result.

It also has two main parts: encoder and decoder. below is desc from paper:

Encoder:

6 layers.each layers has two sub-layers.
the first is multi-head self-attention mechanism;
the second is position-wise fully connected feed-forward network.
for each sublayer. use LayerNorm(x+Sublayer(x)). all dimension=512.

Decoder:

1. The decoder is composed of a stack of N= 6 identical layers.
2. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
attention over the output of the encoder stack.
3. Similar to the encoder, we employ residual connections
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions.  This
masking, combined with fact that the output embeddings are offset by one position, ensures that the
predictions for position i can depend only on the known outputs at positions less than i.

Main Take away from this model:

1) multi-head self attention: use self attention, linear transform multi-times to get projection of key-values, then do ordinary attention; 2) some tricks to improve performance(residual connection,position encoding, poistion feed forward, label smooth, mask to ignore things we want to ignore).

Use this model to do task classification:

Here we only use encode part for task classification, removed resdiual connection, used only 1 layer.no need to use mask. we use multi-head attention and postionwise feed forward to extract features of input sentence, then use linear layer to project it to get logits.

for detail of the model, please check: a2_transformer_classification.py

-------------------------------------------------------------------------

11.Recurrent Entity Network
-------------------------------------------------------------------------
Input:1. story: it is multi-sentences, as context. 2.query: a sentence, which is a question, 3. ansewr: a single label.

Model Structure:

1) Input encoding: use bag of word to encode story(context) and query(question); take account of position by using position mask

   by using bi-directional rnn to encode story and query, performance boost from 0.392 to 0.398, increase 1.5%.

2) Dynamic memory: 

a. compute gate by using 'similiarity' of keys,values with input of story. 

b. get candidate hidden state by transform each key,value and input.

c. combine gate and candidate hidden state to update current hidden state.

3) Output moudle( use attention mechanism):
a. to get possibility distribution by computing 'similarity' of query and hidden state

b. get weighted sum of hidden state using possibility distribution.

c. non-linearity transform of query and hidden state to get predict label.

Main take away from this model:

1) use blocks of keys and values, which is independent from each other. so it can be run in parallel.

2) modelling context and question together. use memory to track state of world; and use non-linearity transform of hidden state and question(query) to make a prediction.

3) simple model can also achieve very good performance. simple encode as use bag of word.

for detail of the model, please check: a3_entity_network.py

under this model, it has a test function, which ask this model to count numbers both for story(context) and query(question). but weights of story is smaller than query.

-------------------------------------------------------------------------

12.Dynamic Memory Network
-------------------------------------------------------------------------
Outlook of Model:

1.Input Module: encode raw texts into vector representation

2.Question Module: encode question into vector representation

3.Episodic Memory Module: with inputs,it chooses which parts of inputs to focus on through the attention mechanism, taking into account of question and previous memory====>it poduce a 'memory' vecotr.

4.Answer Module:generate an answer from the final memory vector.

Detail:

1.Input Module:

  a.single sentence: use gru to get hidden state
  b.list of sentences: use gru to get the hidden states for each sentence. e.g. [hidden states 1,hidden states 2, hidden states...,hidden state n]
  
2.Question Module:
  use gru to get hidden state
  
3.Episodic Memory Module:

  use an attention mechanism and recurrent network to updates its memory. 
     
  a. gate as attention mechanism:
  
     two-layer feed forward nueral network.input is candidate fact c,previous memory m and question q. features get by take: element-wise,matmul and absolute distance of q with c, and q with m.
     
  b.memory update mechanism: take candidate sentence, gate and previous hidden state, it use gated-gru to update hidden state. like: h=f(c,h_previous,g). the final hidden state is the input for answer module.
  
  c.need for multiple episodes===>transitive inference. 
  
  e.g. ask where is the football? it will attend to sentence of "john put down the football"), then in second pass, it need to attend location of john.

4.Answer Module:
take the final epsoidic memory, question, it update hidden state of answer module.

-------------------------------------------------------------------------

TODO 
-------------------------------------------------------------------------------------------------------
1.Character-level Convolutional Networks for Text Classification

2.Convolutional Neural Networks for Text Categorization:Shallow Word-level vs. Deep Character-level

3.Very Deep Convolutional Networks for Text Classification

4.Adversarial Training Methods For Semi-supervised Text Classification

5.Ensemble Models
-------------------------------------------------------------------------


Reference:
-------------------------------------------------------------------------------------------------------
1.Bag of Tricks for Efficient Text Classification

2.Convolutional Neural Networks for Sentence Classification

3.A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification

4.Deep Learning for Chatbots, Part 2 – Implementing a Retrieval-Based Model in Tensorflow, from www.wildml.com

5.Recurrent Convolutional Neural Network for Text Classification

6.Hierarchical Attention Networks for Document Classification

7.Neural Machine Translation by Jointly Learning to Align and Translate

8.Attention Is All You Need

9.Ask Me Anything:Dynamic Memory Networks for Natural Language Processing

10.Tracking the state of world with recurrent entity networks


-------------------------------------------------------------------------

to be continued. for any problem, concat brightmart@hotmail.com
