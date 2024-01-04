"""
Author: Ye Chen ^(=•ェ•=)^
Date: 2023-12-15
Desc: This script is a deep learning model to identify essential proteins.
"""

import os 
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.notebook import trange, tqdm
from tensorflow.keras import layers, Model, optimizers, losses, metrics, Input
from sklearn.metrics import f1_score,confusion_matrix,roc_curve, precision_recall_curve, average_precision_score


# 1 MBIEP model

# ⭐Network embedding process
def _dense_emb(**params):
    units_1 = params["units_1"]
    units_2 = params["units_2"]

    def f(ip):
        dense1 = layers.Dense(units=units_1, activation='relu')(ip)
        dense2 = layers.Dense(units=units_2, activation='relu')(dense1)
        bn = layers.BatchNormalization()(dense2)
        return bn

    return f


# 🍒Subcellular localization process
def _dense_bn_sub(**params):
    units_1 = params["units_1"]
    units_2 = params["units_2"]
    units_3 = params["units_3"]

    def f(ip):
        dense1 = layers.Dense(units=units_1, activation='relu')(ip)
        dense2 = layers.Dense(units=units_2, activation='relu')(dense1)
        bn = layers.BatchNormalization()(dense2)
        dense3 = layers.Dense(units=units_3, activation='relu')(bn)
        return dense3

    return f


# 🍩Gene expression process
def _depth_sep(**params):
    filters = params['filters']
    kernel_size = params['kernel_size']
    activation = params['activation']
    units = params['units']
    pool_size = params['pool_size']
    depth_multiplier = params['depth_multiplier']

    def f(ip):
        # Default channel_last=True
        batch_size, timestep, replicate, channel_num = ip.shape
        c_list = []
        # 1D convolution on each channel
        for i in range(channel_num):
            channel_data = ip[:, :, :, i]
            print(channel_data.shape)
            op = layers.Conv1D(filters=depth_multiplier, kernel_size=kernel_size, activation=activation)(channel_data)
            op = layers.BatchNormalization()(op)
            op = layers.MaxPool1D(pool_size=pool_size)(op)
            c_list.append(op)
        c_layer = layers.Concatenate(axis=-1)(c_list)
        # pointwise process on the concatenated output
        conv11 = layers.Conv1D(filters=filters, kernel_size=1, activation=activation)(c_layer)
        bn = layers.BatchNormalization()(conv11)
        flatten = layers.GlobalMaxPool1D()(bn)
        dense = layers.Dense(units=units, activation=activation)(flatten)
        return dense

    return f


class MBIEP(object):
    @staticmethod
    def build(input_shape_gse, input_shape_emb, input_shape_sub):
        # 🐱 embedding part
        input_emb = Input(shape=input_shape_emb, name='Embedding')
        output_emb = _dense_emb(units_1=16, units_2=16)(input_emb)

        # 🐱 subloc part 
        input_sub = Input(shape=input_shape_sub, name='Subloc')
        output_sub = _dense_bn_sub(units_1=64, units_2=64, units_3=16)(input_sub)

        # 🐱 gse part
        input_gse = Input(shape=input_shape_gse, name='GeneExpression')
        output_gse = _depth_sep(filters=64, kernel_size=2, activation='relu',
                                units=16, pool_size=2, depth_multiplier=64)(input_gse)

        concat = layers.Concatenate(axis=-1)([output_emb, output_sub, output_gse])
        output = layers.Dense(units=1, activation='sigmoid')(concat)

        model = Model(inputs={'Embedding': input_emb, 'Subloc': input_sub,
                              'GeneExpression': input_gse}, outputs=output)
        return model



# 2 Data
class Data(object):
    def __init__(self, j):
        train_eval_rate = 0.8
        # 1. import data
        Embedding = np.load('data/n2v.npy')
        Subloc = np.load('data/' + methods_data[j] + ".npy")
        gse = np.load('data/gse.npy')
        label = np.load('data/label.npy')
   
        # 2. first shuffle
        self.num = len(label)
        label = label.reshape((self.num,1))
        shuffle_index = np.random.permutation(self.num)  
        Embedding = Embedding[shuffle_index]
        Subloc = Subloc[shuffle_index]
        gse = gse[shuffle_index]
        label = label[shuffle_index]
        
        # 3. split train set
        self.trainemb = Embedding[:int(train_eval_rate * self.num)] 
        self.trainsub = Subloc[:int(train_eval_rate * self.num)] 
        self.traingse = gse[:int(train_eval_rate * self.num)] 
        self.trainlabel = label[:int(train_eval_rate * self.num)]
        
        # 4. print information
        print("training data numbers(%d%%): %d" % (train_eval_rate * 100, len(self.trainemb)))
        # 5. strip the pos and neg index
        self.pos_idx = (self.trainlabel == 1).reshape(-1)
        self.neg_idx = (self.trainlabel == 0).reshape(-1) 

        # 6. get the size of train set and print the num of negative and positive amount in the train set
        self.training_size = len(self.trainlabel[self.pos_idx]) * 2
        print("positive data numbers", str(self.training_size // 2))
        print("negative data numbers", str(len(self.neg_idx)))
        
        # 8. split the test set
        self.test_E = Embedding[int((train_eval_rate) * self.num):]
        self.test_S = Subloc[int((train_eval_rate) * self.num):]
        self.test_G = gse[int((train_eval_rate) * self.num):]
        self.test_Y = label[int((train_eval_rate) * self.num):]
        self.test_size = len(self.test_Y)
        
    def shuffle(self):
        # 1. shuffle the negative part
        mark = list(range(int(np.sum(self.neg_idx))))
        np.random.shuffle(mark)

        # 2. even the neg and pos num in the train set
        self.train_E = np.concatenate(
            [self.trainemb[self.pos_idx], self.trainemb[self.neg_idx][mark][:self.training_size // 2]])
        self.train_G = np.concatenate(
            [self.traingse[self.pos_idx], self.traingse[self.neg_idx][mark][:self.training_size // 2]])
        self.train_S = np.concatenate(
            [self.trainsub[self.pos_idx], self.trainsub[self.neg_idx][mark][:self.training_size // 2]])
        self.train_Y = np.concatenate(
            [self.trainlabel[self.pos_idx], self.trainlabel[self.neg_idx][mark][:self.training_size // 2]])
        
        # 3. shuffle the train set concated above
        mark = list(range(self.training_size))
        np.random.shuffle(mark)
        self.train_E = self.train_E[mark]
        self.train_G = self.train_G[mark]
        self.train_S = self.train_S[mark]
        self.train_Y = self.train_Y[mark]


# 3 evaluate
def test_fun(dataset, label, model, j):
      pred = model(dataset, training=False)
      acc = metrics.BinaryAccuracy()(label, pred)
      pre = metrics.Precision()(label, pred)
      rec = metrics.Recall()(label, pred)
      auc = metrics.AUC()(label, pred)
      ap = average_precision_score(label, pred)

      ypred = tf.math.greater(pred, tf.constant(0.5))
      ypred = tf.keras.backend.eval(ypred)
      tn, fp, fn, tp = confusion_matrix(label, ypred).ravel()
      F1 = f1_score(label, ypred)

      fpr, tpr, t = roc_curve(label, pred)
      precision, recall, tr = precision_recall_curve(label, pred)
      print("--->【Test】")
      print('- Accuracy %.4f' % acc)
      print('- Precision %.4f' % pre)
      print('- Recall %.4f' % rec)
      print('- F1-score %.4f' % F1)
      print('- AUC %.4f' % auc)
      print('- AP %.4f' % ap)

# 4 Train
methods_data = ['pca_sub', 'mRMR_sub', 'ica_sub']
methods = ['PCA', 'mRMR', 'ICA']
for j in range(3):
    print("--->【Methods" + methods[j] + "】...")
    print("--->【LoadData】💾...")
    data = Data(j)
    model = MBIEP.build(input_shape_gse=[8, 3, 2], input_shape_emb=[64, ], input_shape_sub=[256, ])
    # set the training parameters
    epochs = 20
    batch_size = 64
    vali_size = 300
    # define loss function and optimizer function
    loss_fun = losses.BinaryCrossentropy(from_logits=False)
    opt_fun = optimizers.Adamax(learning_rate=0.001)

    @tf.function
    def train_fun(dataset, label):
        with tf.GradientTape() as tape:
            pred = model(dataset, training=True)
            loss = loss_fun(label, pred)
        gradient = tape.gradient(loss, model.trainable_variables)
        opt_fun.apply_gradients(zip(gradient, model.trainable_variables))

    print("================begin to train:================")
    for ep in range(epochs):
        # shuffle the data
        data.shuffle()
        # get every batch for training and validation process
        for iter, idx in enumerate(range(0, data.training_size, batch_size)):
            # zip the data together as input
            batch_G = data.train_G[idx:idx + batch_size]
            batch_S = data.train_S[idx:idx + batch_size]
            batch_E = data.train_E[idx:idx + batch_size]
            batch_dict = {'Embedding': batch_E, 'Subloc': batch_S, 'GeneExpression': batch_G}
            batch_Y = data.train_Y[idx:idx + batch_size]
            train_fun(batch_dict, batch_Y)

    test_dict = {'Embedding': data.test_E, 'Subloc': data.test_S, 'GeneExpression': data.test_G}
    test_fun(test_dict, data.test_Y, model, j)

