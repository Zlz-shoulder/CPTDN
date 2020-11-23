# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:41:35 2019

@author: Zhong.Lianzhen
"""
import os
import tensorflow as tf
import numpy as np
#ResNeXt,DenseNet,
from CNN_models import SE_ResNet
from datetime import datetime
from Prepare_data import preprocess
import pandas as pd
from lifelines.utils import concordance_index

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

model_deep = 'model_18'
#Fixed hyper-parameters
dim_interact_feature = 256
momentum = 0.95
num_event = 4
sigma1 = 1.0
num_epochs = 400
img_width = 128
img_heigh = 128

######
#The hyper-parameters were determined using Dataset D3
######
learning_rate_decay = 0.95
learning_rate_base = 1e-3
learning_rate_step = 80
warmup_step = 80

batch_size = 64
keep_prob_rate = 0.6
reg_W_conv = tf.contrib.layers.l2_regularizer(scale=5e-4) #weight regularization for convolutional layers
reg_W = tf.contrib.layers.l2_regularizer(scale=5e-3) #weight regularization for fully connected layers
reg_W_out = tf.contrib.layers.l1_regularizer(scale=1e-2) #weight regularization for output layers

#weights of different losses
alpha = 1.5
belta = 1.0
gamma = 1.0
sita = 1.0
reg_factor = 1.0
intra_loss_weight = [1.0, 2.0]


#MR sequence 
sequence = 'T2' #T1 or T1C
#Path for output
OutputPath = '/YourOutputPath1/'
snapshot_dir = "/YourOutputPath1/T2/64_multi-task_dropout_0.6/checkpoint_path/model_epoch-249"
output_path = os.path.join(OutputPath, sequence)
if not os.path.exists(output_path):
    os.mkdir(output_path)

# "matched_data_from_center2-4.csv"

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    clinic_path = "matched_data_from_center1.csv"
    clinic_msag = pd.read_csv(clinic_path, header = 0, index_col = 0)
    Pat_ID1 = np.array(clinic_msag.index)
    
    clinic_path = "matched_data_from_center2-4.csv"
    clinic_msag = pd.read_csv(clinic_path, header = 0, index_col = 0)
    Pat_ID2 = np.array(clinic_msag.index)
    
    Pat_IDs = np.append(Pat_ID1, Pat_ID2)
    
def _prepare_surv_data(surv_time, surv_event):
    surv_data_y = surv_time * surv_event
    surv_data_y = np.array(surv_data_y, np.float32)
    T = - np.abs(np.squeeze(surv_data_y))
    sorted_idx = np.argsort(T)
    
    return sorted_idx
    


def DeepSurv_loss(surv_time, surv_event, pat_ind, Y_hat):
    # Obtain T and E from self.Y
    # NOTE: negtive value means E = 0
    Y = surv_time * surv_event
    Y_c = tf.squeeze(Y)
    Y_hat_c = tf.squeeze(Y_hat)
    Y_hat_c = tf.gather(Y_hat_c,pat_ind)
    Y_label_T = tf.abs(Y_c)
    Y_label_E = tf.cast(tf.greater(Y_c, 0), dtype=tf.float32)
    Obs = tf.reduce_sum(Y_label_E)
    
    Y_hat_hr = tf.exp(Y_hat_c)
    Y_hat_cumsum = tf.log(tf.cumsum(Y_hat_hr))
    
    # Start Computation of Loss function
    
    # Get Segment from T
    _, segment_ids = tf.unique(Y_label_T)
    # Get Segment_max
    loss_s2_v = tf.segment_max(Y_hat_cumsum, segment_ids)
    # Get Segment_count
    loss_s2_count = tf.segment_sum(Y_label_E, segment_ids)
    # Compute S2
    loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v, loss_s2_count))
    # Compute S1
    loss_s1 = tf.reduce_sum(tf.multiply(Y_hat_c, Y_label_E))
    # Compute Breslow Loss
    loss_breslow = tf.divide(tf.subtract(loss_s2, loss_s1), Obs)
    
    return loss_breslow


def _create_fc_layer(x, output_dim, activation, scope, keep_prob = 1.0, w_reg = None, initial_W = None):
    if initial_W is None:
        initial_W = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(scope):
        layer_out = tf.nn.dropout(x, keep_prob=keep_prob)
        layer_out = tf.layers.dense(inputs=layer_out, use_bias=True, units=output_dim, kernel_initializer=initial_W, kernel_regularizer=w_reg)
        if activation == 'relu6':
            layer_out = tf.nn.relu6(layer_out)
        elif activation == 'relu':
            layer_out = tf.nn.relu(layer_out)
        elif activation == 'tanh':
            layer_out = tf.nn.tanh(layer_out)
        else:
            raise NotImplementedError('activation not recognized')
    
        return layer_out


def Get_loss(output, s_time, s_event, Pat_ind):
    OS_loss_cox = DeepSurv_loss(s_time[:,0], s_event[:,0], Pat_ind[:,0], output[:,0])
    OS_loss_rank = _RankLoss(s_time[:,0], s_event[:,0], Pat_ind[:,0], output[:,0])
    DMFS_loss_cox = DeepSurv_loss(s_time[:,1], s_event[:,1], Pat_ind[:,1], output[:,1])
    DMFS_loss_rank = _RankLoss(s_time[:,1], s_event[:,1], Pat_ind[:,1], output[:,1])
    LRFS_loss_cox = DeepSurv_loss(s_time[:,2], s_event[:,2], Pat_ind[:,2], output[:,2])
    LRFS_loss_rank = _RankLoss(s_time[:,2], s_event[:,2], Pat_ind[:,2], output[:,2])
    DFS_pred = tf.reduce_max(output,1)
    DFS_loss_cox = DeepSurv_loss(s_time[:,3], s_event[:,3], Pat_ind[:,3], DFS_pred)
    DFS_loss_rank = _RankLoss(s_time[:,3], s_event[:,3], Pat_ind[:,3], DFS_pred)
    
    loss_cox = alpha*DFS_loss_cox + belta*OS_loss_cox + gamma*DMFS_loss_cox + sita*LRFS_loss_cox
    loss_rank = alpha*DFS_loss_rank + belta*OS_loss_rank + gamma*DMFS_loss_rank + sita*LRFS_loss_rank
    return loss_cox, loss_rank, DFS_pred

def exponential_decay_with_warmup(warmup_step,learning_rate_base,global_step,learning_rate_step,learning_rate_decay,staircase=False):

    with tf.name_scope("exponential_decay_with_warmup"):
        linear_increase=learning_rate_base*tf.cast((global_step+1)/warmup_step,tf.float32)
        exponential_decay=tf.train.exponential_decay(learning_rate_base,
                                                     global_step-warmup_step,
                                                     learning_rate_step,
                                                     learning_rate_decay,
                                                     staircase=staircase)
        learning_rate=tf.cond(global_step<=warmup_step,
                              lambda:linear_increase,
                              lambda:exponential_decay)
        return learning_rate

def main():
        
    with tf.device('/gpu:0'):
        
        x = tf.placeholder(tf.float32, [None, img_heigh, img_width, 2], name = 'input')
        x1 = tf.placeholder(tf.float32, [None, clinic_num], name = 'input_clinic')
        s_time = tf.placeholder(tf.float32, [None,num_event], name = 'surv_time')
        s_event = tf.placeholder(tf.float32, [None,num_event], name = 'surv_event')
        Pat_ind = tf.placeholder(tf.int32, [None,num_event], name = 'Pat_ind')
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob_rate')
        treatment = tf.placeholder(tf.float32, [None, dim_interact_feature], name = 'treatment')
        global_step = tf.placeholder(tf.int32, [])
        training_flag = tf.placeholder(tf.bool)
        
        pool_avg, _ = SE_ResNet(x, model = model_deep, training = training_flag).model
        fc0 =  _create_fc_layer(pool_avg, dim_interact_feature, 'relu', 'FC_layer1', keep_prob = keep_prob, w_reg = reg_W)
        output = _create_fc_layer(fc0, 3, 'tanh', 'FC_layer1_1', keep_prob = keep_prob, w_reg = reg_W_out)
        fc1 =  _create_fc_layer(pool_avg, dim_interact_feature, 'relu', 'FC_layer2', keep_prob=keep_prob, w_reg = reg_W)
        fc1 = tf.multiply(treatment, fc1)
        output1 = _create_fc_layer(fc1, 1, 'tanh', 'FC_layer2_1', keep_prob = keep_prob, w_reg = reg_W_out)
        
        loss_cox_prog, loss_rank_prog, pred_DFS = Get_loss(output, s_time, s_event, Pat_ind)
        loss_cox_pred = DeepSurv_loss(s_time[:,3], s_event[:,3], Pat_ind[:,3], output1)
        loss_reg = tf.losses.get_regularization_loss()
        g_list = tf.global_variables()
        conv_vars = [g for g in g_list if 'conv' in g.name]
        # for v in conv_vars:
            # print(v)
        # import pdb; pdb.set_trace()
        loss_reg_conv = tf.contrib.layers.apply_regularization(reg_W, weights_list=conv_vars)
        loss_total = intra_loss_weight[0]*loss_cox_prog + intra_loss_weight[1]*loss_cox_pred + reg_factor*loss_reg + 0.05*loss_reg_conv
        # import pdb; pdb.set_trace()
        with tf.name_scope('MIL'):
            instance_index = tf.argmax(pred_DFS, 0)
            pred_value = tf.reduce_max(pred_DFS, 0)
            pred_value = tf.squeeze(pred_value)

        learning_rate = exponential_decay_with_warmup(warmup_step,learning_rate_base,global_step,learning_rate_step,learning_rate_decay,staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = momentum, use_nesterov = True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(loss_total) #,var_list = tf.get_collection("var_fc")

        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)#设置每个GPU使用率0.7代表70%
        #
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            loader = tf.train.Saver(var_list=restore_var)
            loader.restore(sess, snapshot_dir)

            treat = np.array([1]*dim_interact_feature)
            DL_feature = pd.DataFrame()
            for i,Pat_path in enumerate(Pat_IDs):
                Pat_path = str(Pat_path)
                # print(Pat_path)
                instance_batch = preprocess(Pat_path, sequence, False)
                x_n = instance_batch.shape[0]
                IID, rad_fea = sess.run([instance_index, pool_avg], feed_dict = {x: instance_batch, training_flag: False, keep_prob: 1.0, treatment: treat.reshape(1,dim_interact_feature)}) 
                preds = rad_fea[IID,:]
                aFeature = pd.Series(preds)
                aFeature = aFeature.to_frame()
                aFeature.columns = [Pat_path]
                DL_feature = DL_feature.append(aFeature.T)

            # import pdb;pdb.set_trace()
            wb = os.path.join(output_path, sequence + '_dlFeature.csv')
            print(wb)
            DL_feature.to_csv(path_or_buf = wb, encoding='utf-8', index = True, header = True)


if __name__ == '__main__':
    main()