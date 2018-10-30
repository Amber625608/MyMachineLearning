import tensorflow as tf
import numpy as np
import csv
import re
import os
import random
import time
from functools import reduce
import operator
from cnnText import TextCNN
from tensorflow.contrib import learn
import tensorflow.contrib.keras as kr

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

datasPath = r'C:\DL-Data\Amazon'
glovePath =r'C:\Users\hyx\Desktop\深度学习\词向量\glove.6B.50d.txt'
#datasPath = r'F:\深度学习小组\黄叶轩\cnn分类\Amazon 评论数据'
#glovePath =r'F:\深度学习小组\黄叶轩\cnn分类\glove.6B.50d.txt'

files = os.listdir(datasPath)

x_text=[]
y_text=[]

for i_file_all,file in enumerate(files):
    pass
i_file_all+=1
for i_file,file in enumerate(files):
    dataPath=datasPath+"\\"+file
    with open(dataPath, 'r',encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        reviewText_row = [row['reviewText'] for row in reader]
    print("【加载完成】",file)
    reviewText_clear = [s.strip() for s in reviewText_row]#去头尾空格
    x_text += reviewText_clear

    y_target=[0]*i_file_all
    y_target[i_file]=1
    y_text +=[y_target]*len(reviewText_clear) #生成目标值
    reviewText_clear.clear()
print("【加载中】正在清洗数据")
x_text = [clean_str(sent) for sent in x_text]


#截断数据
jieduan_data=80000
x_text=x_text[0:jieduan_data]
y_text=y_text[0:jieduan_data]
''''''

x_text_size=len(x_text)
print("【加载完成】","导入所有评论")
print("【统计】","评论总数：",len(y_text))

#把句子换成单词
x_text_word=[x.split(" ") for x in x_text]
print("x_text_word is done")
x_text_word_one=reduce(operator.add, x_text_word)#转换成一维
print("x_text_word_one is done")

#读取词向量
def loadGloVe(glovePath):
    vocab = []
    embd = []
    vocab_num = []
    vocab.append('unk') #装载不认识的词
    embd.append([0]*50) #这个emb_size可能需要指定
    vocab_num.append(0)
    file = open(glovePath,'r',encoding='utf-8')
    line_i=1#词向量数
    for i,line in enumerate(file.readlines()):
        row = line.strip().split(' ')
        # 生成自己的小词库
        if(row[0] not in x_text_word_one):
            continue
        vocab.append(row[0])
        embd.append(row[1:])
        vocab_num.append(line_i)
        line_i+=1
        if i%10000==0:
            print("【加载中】",i,'/400000')
    print("【加载完成】",'Glove词向量导入完成')
    file.close()
    return vocab,embd,vocab_num
vocab,embd,vocab_num = loadGloVe(glovePath)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
dic = dict(map(lambda x,y:[x,y],vocab,embedding))
dic_num = dict(map(lambda x,y:[x,y],vocab,vocab_num))
print("【加载完成】字典生成完毕")
'''with open(glovePath,'r',encoding='utf-8') as EmbedFile:
    dict_WE={}
    list_WE=[]
    for i,line in enumerate(EmbedFile):
        list_WE=line.split()
        list_1=list_WE[0]
        list_2=list(map(eval, list_WE[1:]))
        dict_WE.update({list_1:list_2})
        if(i%10000==0):
            print("读取词向量中:",i)
print("dict_WE done")


data_max_size=20000#截断数据
#将词向量和源数据匹配
x_input=[]
max_len=0
for i,review in enumerate(x_text_list):
    x_view=[]
    for j,danci in enumerate(review):
        try:
            x_view.append(dict_WE[danci])
        except KeyError:
            #x_view.append(tf.truncated_normal([50],mean=0.0,stddev=1.0))
            x_view.append([0]*50)
        finally:
            if(j>max_len):
                max_len=j
    x_input.append(x_view)
    if(i%10000==0):
        print("匹配中:",i,"/",x_text_size)
    if(i==(data_max_size-1)):
        break
y_input=y_text[0:data_max_size]
print("pipei is done")'''

'''#每个句子的最大单词数
max_document_length = max_len+1#这个+1 太坑了找了半天
print("最大长度:",max_document_length)
print("here is wrong?")
x_input_max = list(map(lambda l:l + [[0]*50]*(max_document_length - len(l)), x_input))#这个地方效率太低了，贼慢
print("句子转换成相同长度 done")
#print("x_input_max:",x_input_max[data_max_size-1])'''

'''#我打算把输入的数组拆成很多很多份，解决内存溢出的问题
def list_of_groups(init_list, children_list_len):
    list_of_groups = zip(*(iter(init_list),) *children_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % children_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list
chaifenshu=100  #拆分数
x_input_max_all=list_of_groups(x_input_max,int(len(x_input_max)/chaifenshu))
x_input_array=np.array(x_input_max_all[0])
for  i,x_input_max_ in enumerate(x_input_max_all):
    if (i == 0):
        continue
    x_input_array = np.concatenate((x_input_array, x_input_max_), axis=0)
    del x_input_max_[:]
    if(i%10==0):
        print("转换为np.array中：",i,"/",chaifenshu)

#将句子转换为多个单词
x_text_list=[]
for i,review in enumerate(x_text):
    x_view_list=[]
    for j,danci in enumerate(review.split(' ')):
        x_view_list.append(danci)

    x_text_list.append(x_view_list)
print("x_text_list done")

#建立词和序号的索引表  key是序号  value是单词
dict_word={}
danci_i=1
for i,review in enumerate(x_text):
    for j,danci in enumerate(review):
        if(danci in dict_word.values()):
            continue
        else:
            dict_word.update({danci_i:danci})
            danci_i+=1
print("dict_word done")'''
#print(x_text)
#把句子换成单词，再把单词换成对相应序号数，再填充至最大
x_xuhao=[]

#x_text_word=[x.split(" ") for x in x_text]
del x_text

for i,juzi in enumerate(x_text_word):
    x_xuhao_juzi = []
    for j,word in enumerate(juzi):
        if(word in dic_num):
            x_xuhao_juzi.append(dic_num[word])
        else:
            x_xuhao_juzi.append(0)
    x_xuhao.append(x_xuhao_juzi)

    if(i%20000==0):
        print("【加载中】单词转序号：",i,"/",x_text_size)

del x_text_word
max_document_length_list=[len(x) for x in x_xuhao]
print(max_document_length_list)
max_document_length = max(max_document_length_list)#每个句子的最大单词数
def Get_Average(list):
   sum = 0
   for item in list:
      sum += item
   return sum/len(list)

ave_document_length =Get_Average(max_document_length_list)
del max_document_length_list

print("平均数:",ave_document_length)
most_document_length=int(ave_document_length/0.25)#大约满足8成的数
print("满足大部分长度的数:",most_document_length)
#max_document_length=500
x_xuhao_max=tf.contrib.keras.preprocessing.sequence.pad_sequences(x_xuhao,maxlen=most_document_length,padding='post',truncating='post',value = 0)
del x_xuhao

print("【加载完成】","设为相同长度完成")



'''
#每句评论长度都设为最大值
max_document_length = max([len(x.split(" ")) for x in x_text])#每个句子的最大单词数
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
print("here")
tobelist=vocab_processor.fit_transform(x_text)
print("tobelist is done")
list_z=list(tobelist)#这里内存被榨干
print("list_z is done")

for i,pinglun in enumerate(list_z):
    if(len(pinglun)!=max_document_length):
        print("错误的数据：",i,",",len(pinglun))
        print(pinglun)
print("检测完成")
'''


x_input_array = np.array(x_xuhao_max)
del x_xuhao_max
print("【加载完成】","转换为np.array完成")


#list_lenth=max_document_length
'''
#生成一张自己词向量索引表
all_word_size=len(vocab_processor.vocabulary_)
initW= [[0]*50 for x in range(0, all_word_size)]
print("【统计】""所有单词的个数：",len(vocab_processor.vocabulary_))
for i,w in enumerate(vocab_processor.vocabulary_._mapping):
    #tensorflow生成的词表
        arr = []

        if w in vocab:
            try:
                arr = dic[w]
            except IndexError:
                print("错误的数据:",w,dic[w])
            #pre_emd,我们预训练的词向量表
            if len(arr) > 0:
                idx = vocab_processor.vocabulary_.get(w)
                #print("idx:",idx)
                #print("arr:",arr.astype(np.float32))
                initW[idx] = arr.astype(np.float32)#np.asarray(arr).astype(np.float32)

        if(i%3000==0):
            print("生成词向量索引表中",i,"/",all_word_size)

print("【加载完成】","词向量索引表建立完成")
'''

'''#print(list_z)
#print("开始检查是否不一致")
#检查格式是否错误
geshi=0
for i,geshi_i in enumerate(list_z):
    if(len(geshi_i)!=geshi):
        print("不匹配:",len(geshi_i),",",i)
        geshi=len(geshi_i)

print("检查结束")'''


#这个地方也贼慢

y_input_array=np.array(y_text)
print("【统计】""输入数据的维度:",x_input_array.shape)
#print("shape of x:",x_input_array.shape)


# 数据打乱

shuffle_indices = np.random.permutation(np.arange(len(y_text))) #np.random.permutation 是从0开始的

rand_y=y_input_array[shuffle_indices]
rand_x=x_input_array[shuffle_indices]

print("【加载完成】","数据已打乱")


#分割训练数据和测试数据
fenge=int(len(rand_y)*0.9)
x_train,x_test=rand_x[:fenge],rand_x[fenge:]
y_train,y_test=rand_y[:fenge],rand_y[fenge:]

#print("【统计】","不重复的单词总数:",len(vocab_processor.vocabulary_))
#训练的模型
print("【统计】","目标分类数:",i_file_all)
cnn = TextCNN(
                sequence_length=most_document_length, #每个句子里面的单词的数量长度
                num_classes=i_file_all, #分为几类
                vocab_size=vocab_size,
                embedding_size=50,
                filter_sizes=[3,4,5], # 上面定义的filter_sizes拿过来，"3,4,5"按","分割
                num_filters=128# 一共有几个filter

                )# l2正则化项

print("【加载完成】","模型加载完成")
#训练优化
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-4)
grads_and_vars = optimizer.compute_gradients(cnn.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

sess=tf.Session()
sess.run(tf.global_variables_initializer())


def train_step(x_batch, y_batch):

    feed_dict = {
        cnn.dropout_keep_prob: 0.5,
        cnn.embedding_placeholder:embedding,
        cnn.input_x: x_batch,
        cnn.input_y: y_batch

    }
    #print("train_step  feed_dict is done")
    since = time.time()
    _, step, loss, accuracy,eb = sess.run(
        [train_op, global_step,cnn.loss,cnn.accuracy,cnn.embedding_init],
        feed_dict)
    time_elapsed = time.time() - since
    print(" step {}, loss {:g}, acc {:g}".format( step, loss, accuracy))
    print("time:",time_elapsed)

def dev_step(x_batch, y_batch):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0,
        cnn.embedding_placeholder: embedding,
    }
    step, loss, accuracy,eb = sess.run(
        [global_step, cnn.loss, cnn.accuracy,cnn.embedding_init],
        feed_dict)

    print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


train_test_num=20  #每训练多少次测试一次
train_batch_size=64 #每次训练的大小
test_batch_size=256  #每次测试的大小
test_random_num=int(len(x_test)/test_batch_size)-1  #测试的轮数
print("test_random_num:",test_random_num)

batches = batch_iter(list(zip(x_train, y_train)),train_batch_size, 1)
#batches_test = batch_iter_test(list(zip(x_test, y_test)),test_batch_size, test_batch_num)
print("【头大】","开始训练")
for batch in batches:
    x_batch, y_batch = zip(*batch)
    train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    if current_step % train_test_num == 0:
        print("\nEvaluation:")
        start_test=(random.randint(0,test_random_num)*test_batch_size)   #随机出的测试起始位置
        end_test=start_test+test_batch_size
        x_test_batch,y_test_batch=x_test[start_test:end_test],y_test[start_test:end_test]
        dev_step(x_test_batch, y_test_batch)#一次载入的数据太多?
        print("")

    if(current_step==1000):
        pass
        #break

print("【结束】","All done")