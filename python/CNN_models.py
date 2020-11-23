# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:45:41 2020

@author: ZhongLianzhen
"""

import tensorflow as tf
slim = tf.contrib.slim

CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
def batch_normalization(x, training, name):
    with arg_scope([batch_norm],
                   scope = name,
                   updates_collections = None,
                   decay = 0.9,
                   center = True,
                   scale = True,
                   zero_debias_moving_mean = False) :
        return tf.cond(training,
                       lambda : batch_norm(inputs = x, is_training = training, reuse = None),
                       lambda : batch_norm(inputs = x, is_training = training, reuse = True))
def conv_layer(input_data, filter_num, kernel, stride = 1, use_bias = False, padding='SAME', layer_name="conv"):
    network = tf.layers.conv2d(inputs = input_data, filters = filter_num, kernel_size = kernel, strides=stride, use_bias = use_bias, name = layer_name, padding=padding)
    return network

def Average_pooling(x, pool_size=[2,2], stride = 2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs = x, pool_size = pool_size, strides = stride, padding = padding)


def Relu(x):
    return tf.nn.relu(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)


def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Fully_connected(x, units, layer_name='fully_connected') :
        return tf.layers.dense(inputs=x, use_bias=False, units=units, name = layer_name)
    
def BasicResidualSEBlock(x, scope, is_training, out_channels, stride, r=4):
    expansion = 1
    in_channels = x.shape.as_list()[-1]
    with tf.variable_scope(scope):
        with tf.variable_scope('shortcut'):
            if stride != 1 or in_channels != out_channels * expansion:
                shortcut = conv_layer(x, filter_num = out_channels * expansion, kernel=[1,1], stride=stride, layer_name = 'conv1')
                shortcut = batch_normalization(shortcut, training=is_training, name='bn')
            else:
                shortcut = x
        with tf.variable_scope('residual'):
            residual = conv_layer(x, filter_num = out_channels, kernel=[3,3], stride=stride, layer_name = 'conv3_1')
            residual = batch_normalization(residual, training=is_training, name='bn_1')
            residual = Relu(residual)
            residual = conv_layer(residual, filter_num = out_channels * expansion, kernel=[3,3], stride=1, layer_name = 'conv3_2')
            residual = batch_normalization(residual, training=is_training, name='bn_2')
            residual = Relu(residual)
        with tf.variable_scope('squeeze_layer'):

            squeeze = tf.math.reduce_mean(residual, [1, 2], name='GAP', keepdims=False)

            excitation = Fully_connected(squeeze, units = out_channels * expansion // r, layer_name = 'fc1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units = out_channels * expansion, layer_name = 'fc2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_channels * expansion])
            scale = residual * excitation
        x = Relu(shortcut + scale)
        
    return x
def transform_layer(x, out_channels, stride, training, scope):
        with tf.variable_scope(scope):
            x = conv_layer(x, filter_num = out_channels, kernel=[1,1], stride=1, layer_name='conv1')
            x = batch_normalization(x, training=training, name='bn1')
            x = Relu(x)

            x = conv_layer(x, filter_num = out_channels, kernel=[3,3], stride=stride, layer_name='conv2')
            x = batch_normalization(x, training=training, name='bn2')
            x = Relu(x)
            return x 
def BottleneckResidualSEBlock(x, scope, is_training, out_channels, stride, r=4):
    expansion = 4
    in_channels = x.shape.as_list()[-1]
    with tf.variable_scope(scope):
        with tf.variable_scope('shortcut'):
            if stride != 1 or in_channels != out_channels * expansion:
                shortcut = conv_layer(x, filter_num = out_channels * expansion, kernel=[1,1], stride=stride, layer_name = 'conv1')
                shortcut = batch_normalization(shortcut, training=is_training, name='bn')
            else:
                shortcut = x
        with tf.variable_scope('group_residual'):
            C = CARDINALITY
            D = DEPTH * out_channels // BASEWIDTH
            layers_split = list()
            for i in range(C):
                splits = transform_layer(x, D, stride = stride, training=is_training, scope = 'splitN_' + str(i))
                layers_split.append(splits)
            residual = tf.concat(layers_split,axis = 3)
            residual = conv_layer(residual, filter_num = out_channels * expansion, kernel=[1,1], stride=1, layer_name = 'trans_conv')
            residual = batch_normalization(residual, training=is_training, name='trans_conv_bn')
#            residual = Relu(residual)
        with tf.variable_scope('squeeze_layer'):

            squeeze = tf.math.reduce_mean(residual, [1, 2], name='GAP', keepdims=False)

            excitation = Fully_connected(squeeze, units = out_channels * expansion // r, layer_name = 'fc1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units = out_channels * expansion, layer_name = 'fc2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_channels * expansion])
            scale = residual * excitation
        x = Relu(shortcut + scale)
        
    return x

class SE_ResNet():
    def __init__(self, x, model, training, shuffle = True, reduction_ratio = 4):
        self.reduction_ratio = reduction_ratio
        self.is_training = training
        self.shuffle = shuffle
        self.end_points_collection = model + '_end_points'
        self.model = self.Build_SE_ResNext(x, model)
        
    def first_layer(self, input_x, scope = 'first_layer'):
        with tf.variable_scope(scope):
            x = conv_layer(input_x, filter_num = 32, kernel=[5, 5], stride=2, layer_name='conv5')
            x =  batch_normalization(x, training=self.is_training, name='bn1')
            x = Relu(x)
            
            x = conv_layer(x, filter_num = 64, kernel=[3, 3], stride=1, layer_name='conv3')
            x =  batch_normalization(x, training=self.is_training, name='bn2')
            x = Relu(x)
            
            x = Average_pooling(x)
    
            
        return x

    def SE_residual_basic(self, input_x, out_dim, layer_name, res_block):
        
        for i in range(res_block):
            if i == 0:
                input_x = BasicResidualSEBlock(x=input_x, scope=layer_name + '_' + str(i), is_training=self.is_training, out_channels=out_dim, stride=2, r=self.reduction_ratio)
            else:
                input_x = BasicResidualSEBlock(x=input_x, scope=layer_name + '_' + str(i), is_training=self.is_training, out_channels=out_dim, stride=1, r=self.reduction_ratio)
                
        return input_x

    def SE_residual_bottleneck(self, input_x, out_dim, layer_name, res_block):
        
        for i in range(res_block):
            if i == 0: 
                input_x = BottleneckResidualSEBlock(x=input_x, scope=layer_name + '_' + str(i), is_training=self.is_training, out_channels=out_dim, stride=2, r=self.reduction_ratio)
            else:
                input_x = BottleneckResidualSEBlock(x=input_x, scope=layer_name + '_' + str(i), is_training=self.is_training, out_channels=out_dim, stride=1, r=self.reduction_ratio)
                
        return input_x       
    def Build_SE_ResNext(self, input_x, model):
        
        input_x = self.first_layer(input_x, scope = 'first_layer')
        input_x = slim.utils.collect_named_outputs(self.end_points_collection, 'first_layer', input_x)
        if model == 'model_18':
            x = self.SE_residual_basic(input_x, out_dim = 128, layer_name = 'stage_1', res_block = 2)
            x = slim.utils.collect_named_outputs(self.end_points_collection, 'stage_1', x)
            x = self.SE_residual_basic(x, out_dim = 256, layer_name = 'stage_2', res_block = 2)
            x = slim.utils.collect_named_outputs(self.end_points_collection, 'stage_2', x)
            x = self.SE_residual_basic(x, out_dim = 512, layer_name = 'stage_3', res_block = 2)
            x = slim.utils.collect_named_outputs(self.end_points_collection, 'stage_3', x)
        elif model == 'model_50':
            x = self.SE_residual_bottleneck(input_x, out_dim = 128, layer_name = 'stage_1', res_block = 3)
            x = slim.utils.collect_named_outputs(self.end_points_collection, 'stage_1', x)
            x = self.SE_residual_bottleneck(x, out_dim = 256, layer_name = 'stage_2', res_block = 4)
            x = slim.utils.collect_named_outputs(self.end_points_collection, 'stage_2', x)
            x = self.SE_residual_bottleneck(x, out_dim = 512, layer_name = 'stage_3', res_block = 3)
            x = slim.utils.collect_named_outputs(self.end_points_collection, 'stage_3', x)
        else:
            raise NotImplementedError
        
        x = tf.math.reduce_mean(x, [1, 2], name='GAP', keepdims=False)
        x = slim.utils.collect_named_outputs(self.end_points_collection, 'GAP', x)
        end_points = slim.utils.convert_collection_to_dict(
              self.end_points_collection)

        return x, end_points
    
# if __name__ == "__main__":
    # x = tf.placeholder(tf.float32, [None, 128, 128, 2])

    # net, end_points = SE_ResNet(x, model = 'model_50', training = True).model
    # for i in end_points:
        # print(i, end_points[i])
    # for v in tf.trainable_variables():
        # print(v.name)
#    import pdb; pdb.set_trace()
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
#    
#    # Saver for storing checkpoints of the model.
#    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)
#    
#    # Load variables if the checkpoint is provided.
#    if args.ckpt > 0 or args.restore_from is not None or args.imagenet is not None:
#        loader = tf.train.Saver(var_list=restore_var)
#        loader.restore(sess, ckpt_path)