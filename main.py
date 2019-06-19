# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:50:04 2019

@author: Administrator
"""

import os
import glob
import time
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def read_data(path):
    
    cate = [path + '\\' + x for x in os.listdir(path) if 
            os.path.isdir(path + '\\' + x)]
    data = []
    label = []
    
    for idx, folder in enumerate(cate):
        for t in glob.glob(folder + '\*.mat'):
            print('reading the data:%s'%(t))
            mat = loadmat(t)
            mat_data = np.append(mat['Var_1'], mat['Var_2'], axis = 1)
            mat_data = np.append(mat_data, mat['Var_3'], axis = 1)
            
            data.append(mat_data)
            label.append(np.array([idx for _ in range(len(mat_data))],
                                   dtype = np.int))
            
    return np.array(data), np.array(label)

def data_preprocessing(data, label):
    
    x_train, x_test, y_train, y_test = train_test_split(data[0], 
                                                        label[0],
                                                        test_size = 0.5)
    
    for i in range(1, len(data)):
        _x_train, _x_test, _y_train, _y_test = train_test_split(data[i], 
                                                                label[i],
                                                                test_size = 0.5)
        x_train = np.append(x_train, _x_train, axis = 0)
        y_train = np.append(y_train, _y_train)
        x_test = np.append(x_test, _x_test, axis = 0)
        y_test = np.append(y_test, _y_test)
    
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)
    
    return x_train, x_test, y_train, y_test

def minibatches(inputs = None, targets = None, batch_size = None, shuffle = False):
        
        assert len(inputs) == len(targets)
        
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]

if __name__ == '__main__':
    
    data, label = read_data(r'.\data')
    
    x_train, x_test, y_train, y_test = data_preprocessing(data, label)
    x_train = x_train[:, :, np.newaxis]
    x_test = x_test[:, :, np.newaxis]
    
    #-----------------构建网络----------------------
    x = tf.placeholder(tf.float32, shape=[None, 30, 1], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None,], name='y_')
    
    #第一个卷积层(30->15)
    conv1 = tf.layers.conv1d(
            inputs = x,
            filters = 10,
            kernel_size = 5,
            padding = "same",
            activation = tf.nn.relu,
            kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01))
    
    pool1 = tf.layers.max_pooling1d(inputs = conv1, pool_size = 2, strides = 2)
    
    re1 = tf.reshape(pool1, [-1, 15 * 10])
    
    #全连接层
    dense1 = tf.layers.dense(inputs = re1, 
                             units = 300, 
                             activation = tf.nn.relu,
                             kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01),
                             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))
    
    logits= tf.layers.dense(inputs = dense1, 
                            units = 5, 
                            activation = None,
                            kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01),
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))
    
#    y = tf.nn.softmax(logits)
    #---------------------------网络结束---------------------------
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels = y_, logits = logits)
    train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)    
    acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #训练和测试数据
 
    n_epoch = 100
    batch_size = 60
    
    sess=tf.InteractiveSession()  
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epoch):
        
        start_time = time.time()
        
        #training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _,err,ac=sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err; train_acc += ac; n_batch += 1
    
        print(epoch)
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))
        
        #testing
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_test, y_test, batch_size, shuffle=False):
            err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err; val_acc += ac; n_batch += 1
        print("   validation loss: %f" % (val_loss/ n_batch))
        print("   validation acc: %f" % (val_acc/ n_batch))
    
    sess.close()
    
    
    
    