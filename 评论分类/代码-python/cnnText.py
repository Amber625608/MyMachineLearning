import tensorflow as tf
import numpy as np
import csv
import re
import os
from tensorflow.contrib import learn

class TextCNN(object):
    def __init__(
            self, sequence_length, num_classes,vocab_size,
            embedding_size, filter_sizes, num_filters):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes ], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)
        l2_reg_lambda = 0.0
        embedding_dim=50
        '''#匹配词向量

        self.x_input_we = []
        sess_zh=tf.Session()
        print("启动1")
        sess_zh.run(fetches=tf.global_variables_initializer())
        print("启动2")
        input_x_np = self.input_x.eval(session=sess_zh)
        print("匹配1")
        for review in input_x_np:
            print("匹配2")

            x_view = []
            for j, danci in enumerate(review):  #这里的单词是序号
                try:
                    if(danci==0):
                        print("匹配0")
                        x_view.append([0] * 50)
                        continue
                    print("匹配3")
                    x_view.append(dict_WE[dict_word[danci]])#dict_WE[dict_word[danci]]
                except KeyError:
                    # x_view.append(tf.truncated_normal([50],mean=0.0,stddev=1.0))
                    x_view.append([0] * 50)

            self.x_input_we.append(x_view)
        np_x_input_we=np.array(self.x_input_we)
        self.tf_x_input_we = tf.convert_to_tensor(np_x_input_we)
        sess_zh.close()
        print("匹配done")'''
        #embeding
        self.W = tf.Variable(tf.constant(0.0,shape=[vocab_size, embedding_dim]),
                        trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        self.embedding_init = self.W.assign(self.embedding_placeholder)
        #print("embedding_init is done")
        self.embedded_chars=tf.nn.embedding_lookup(self.W, self.input_x)
        #print("embedded_chars is done")
        self.x_expanded = tf.expand_dims(self.embedded_chars, -1)  # 增加一个维度（通道数）
        #print("维度加一")

        pooled_outputs = []


        for i, filter_size in enumerate(filter_sizes):

            filter_shape = [filter_size, embedding_size, 1, num_filters]  # 50:维，1：通道，128：窗口个数
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

            conv = tf.nn.conv2d(
                self.x_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")

            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],  # (h-filter+2padding)/strides+1=h-f+1
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")

            pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)  # 损失函数为什么要在这里导入
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss


        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
