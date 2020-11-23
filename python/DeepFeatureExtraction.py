# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:41:35 2020

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
output_path = os.path.join(OutputPath, sequence)
if not os.path.exists(output_path):
    os.mkdir(output_path)

output_path = os.path.join(output_path, '64_multi-task_dropout_0.6')
checkpoint_path = os.path.join(output_path, "checkpoint_path")
if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
log_path = os.path.join(output_path,'logs.txt')
clinic_path = "matched_data_from_center1.csv"
#set No. of patients with event : No. of censored patients as 0.7
sampling_rate = 0.7 #Sampling ratio
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    clinic_msag = pd.read_csv(clinic_path, header = 0, index_col = 0)
    #Dataset D1
    tra_msag = clinic_msag[clinic_msag['data_cohort']==1]
    #Dataset D3
    val_msag = clinic_msag[clinic_msag['data_cohort1']==1]
    #Internal test cohort
    test_msag = clinic_msag[clinic_msag['data_cohort2']==1]

    tra_Pat_ID = np.array(tra_msag.index)
    tra_treat = np.array(tra_msag.loc[:, 'treatment'], np.float32)
    tra_time = np.array(tra_msag.loc[:, ['OS.time', 'DMFS.time', 'LRRFS.time','DFS.time']], np.float32)
    tra_event = np.array(tra_msag.loc[:, ['OS', 'DMFS', 'LRRFS','DFS']], np.float32)
    tra_FFS_time = np.array(tra_msag.loc[:, 'DFS.time'], np.float32)
    tra_FFS_event = np.array(tra_msag.loc[:, 'DFS'], np.float32)
    tra_FFS_event[tra_FFS_event<0.0] = 0.0
    clinic_factors = np.array(tra_msag.loc[:, clinic_vars], np.float32)


    val_Pat_ID = np.array(val_msag.index)
    val_treat = np.array(val_msag.loc[:, 'treatment'], np.float32)
    val_FFS_time = np.array(val_msag.loc[:, 'DFS.time'], np.float32)
    val_FFS_event = np.array(val_msag.loc[:, 'DFS'], np.float32)
    val_FFS_event[val_FFS_event<0.0] = 0.0
    clinic_factors_val = np.array(val_msag.loc[:, clinic_vars], np.float32)

    test_Pat_ID = np.array(test_msag.index)
    test_treat = np.array(test_msag.loc[:, 'treatment'], np.float32)
    test_FFS_time = np.array(test_msag.loc[:, 'DFS.time'], np.float32)
    test_FFS_event = np.array(test_msag.loc[:, 'DFS'], np.float32)
    test_FFS_event[test_FFS_event<0.0] = 0.0
    clinic_factors_test = np.array(test_msag.loc[:, clinic_vars], np.float32)

    # import pdb; pdb.set_trace()
    ind_all = np.array(range(len(tra_msag)))
    ind_0 = ind_all[tra_msag.loc[:,'DFS'] == -1]
    ind_1 = ind_all[tra_msag.loc[:,'DFS'] == 1]
    nn_1 = len(ind_1)
    nn_0 = len(ind_0)
    print('number of patients without the event befor sampling: %d' % nn_0)
    print('number of patients with the event befor sampling: %d' % nn_1)
    np.random.seed(0)
    if nn_1 < sampling_rate*nn_0:
        out = np.random.choice(len(ind_1),int(sampling_rate*nn_0))
        ind_1 = ind_1[out]
    else:
        out = np.random.choice(len(ind_0),int(nn_1/sampling_rate))
        ind_0 = ind_0[out]

    r0 = int(batch_size/(1+sampling_rate))
    r1 = batch_size - r0
    print('number of patients without the event after sampling: %d' % len(ind_0))
    print('number of patients with the event after sampling: %d' % len(ind_1))

    num_batchs = min(len(ind_0)//r0, len(ind_1)//r1)
    # print([r0,r1])
    print('iterations per epoch: %d' % num_batchs)
    
def _prepare_surv_data(surv_time, surv_event):
    surv_data_y = surv_time * surv_event
    surv_data_y = np.array(surv_data_y, np.float32)
    T = - np.abs(np.squeeze(surv_data_y))
    sorted_idx = np.argsort(T)
    
    return sorted_idx
    
def GetData(ind0, ind1):
    ind = np.hstack((ind0,ind1))
    np.random.shuffle(ind)
    input_idx = np.zeros((batch_size, 4), dtype = np.int32)
    
    input_time = tra_time[ind]
    input_event = tra_event[ind]
    
    sorted_idx = _prepare_surv_data(input_time[:,0], input_event[:,0])
    input_idx[:,0] = sorted_idx
    input_time[:,0] = input_time[sorted_idx,0]
    input_event[:,0] = input_event[sorted_idx,0]
    
    sorted_idx = _prepare_surv_data(input_time[:,1], input_event[:,1])
    input_idx[:,1] = sorted_idx
    input_time[:,1] = input_time[sorted_idx,1]
    input_event[:,1] = input_event[sorted_idx,1]
    
    sorted_idx = _prepare_surv_data(input_time[:,2], input_event[:,2])
    input_idx[:,2] = sorted_idx
    input_time[:,2] = input_time[sorted_idx,2]
    input_event[:,2] = input_event[sorted_idx,2]
    
    sorted_idx = _prepare_surv_data(input_time[:,3], input_event[:,3])
    input_idx[:,3] = sorted_idx
    input_time[:,3] = input_time[sorted_idx,3]
    input_event[:,3] = input_event[sorted_idx,3]
    
    input_x1 = clinic_factors[ind, :]
    
    treat = 0.5*tra_treat[ind]
    treat = treat.reshape((-1, 1))
    treat_out = np.ones((1, dim_interact_feature))
    treat_out = treat * treat_out

    tmp = tra_Pat_ID[ind]
    Pat_path = [str(i) for i in tmp]
    
    return treat_out, Pat_path, input_x1, input_time, input_event, input_idx
    


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

    DFS_pred = tf.py_func(np.max, [output,1], tf.float32)
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
        loss_reg_conv = tf.contrib.layers.apply_regularization(reg_W_conv, weights_list=conv_vars)
        loss_total = intra_loss_weight[0]*loss_cox_prog + intra_loss_weight[1]*loss_cox_pred + reg_factor*loss_reg + loss_reg_conv
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

        restore_var = [v for v in tf.trainable_variables()]
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        restore_var += bn_moving_vars
        saver = tf.train.Saver(max_to_keep = 30)
        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)
        #
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            
            gsp = 0
            best_val_C_index = 0.0
            # Loop over number of epochs
            for epoch in range(num_epochs):
            
                # print("{} Start epoch number: {}".format(datetime.now(), epoch))
                np.random.shuffle(ind_0)
                np.random.shuffle(ind_1)
                # Initialize iterator with the training dataset
                train_risk = 0.0
                prog_risk = 0.0
                pred_risk = 0.0
                reg_risk = 0.0
                conv_risk = 0.0
                # import pdb;pdb.set_trace()
                for i in range(num_batchs):
                    gsp += 1
                    ind0 = ind_0[i*r0:(i+1)*r0]
                    ind1 = ind_1[i*r1:(i+1)*r1]
                    treat, img_path, input_x1, input_time, input_event, input_idx = GetData(ind0,ind1)
                    vd_img = []
                    for i,Pat_path in enumerate(img_path):
                        instance_batch = preprocess(Pat_path, sequence, True)
                        x_n = instance_batch.shape[0]
                        IID = sess.run(instance_index, feed_dict = {x: instance_batch, x1: np.tile(input_x1[i].reshape(1,clinic_num),(x_n,1)), keep_prob: 1.0, training_flag: False, treatment: treat[i].reshape(1,dim_interact_feature)})
                        vd_img.append(instance_batch[IID,:,:,:])
                    img_batch = np.stack(vd_img)
                    _, conv_ls, reg_ls, prog_ls, pred_ls, total_ls, now_lr = sess.run([train_step, loss_reg_conv, loss_reg, loss_cox_prog, loss_cox_pred, loss_total, learning_rate], feed_dict = {global_step:gsp, treatment: treat, x: img_batch, x1: input_x1, s_time: input_time, s_event: input_event, Pat_ind: input_idx, training_flag: True, keep_prob: keep_prob_rate})
                    conv_risk += conv_ls
                    reg_risk += reg_ls
                    train_risk += total_ls
                    prog_risk += prog_ls
                    pred_risk += pred_ls

                conv_risk /= num_batchs
                reg_risk /= num_batchs
                train_risk /= num_batchs
                prog_risk /= num_batchs
                pred_risk /= num_batchs
                line = 'epoch: %d, learning rate: %.5f, tatol_loss: %.4f, conv_reg_loss: %.4f, reg_loss: %.4f, prognosis-cox loss: %.4f, predict-cox loss: %.4f' % (epoch + 1, now_lr, train_risk, conv_risk, reg_risk, prog_risk, pred_risk)
                print(line)
                with open(log_path, 'a') as f:
                    f.write(line + '\n')
                #evaluate model in Dataset D3
                val_pred = []
                for i, Pat_path in enumerate(val_Pat_ID):
                    Pat_path = str(Pat_path)
                    instance_batch = preprocess(Pat_path, sequence, False)
                    x_n = instance_batch.shape[0]
                    xd = val_treat[i]
                    treat = np.array([xd]*dim_interact_feature)
                    Pat_pred = sess.run(pred_value, feed_dict = {x: instance_batch, x1: np.tile(clinic_factors_val[i,:].reshape(1,clinic_num),(x_n,1)), training_flag: False, keep_prob: 1.0, treatment: treat.reshape(1,dim_interact_feature)}) 
                    val_pred.append(-np.exp(Pat_pred))

                val_pred = np.array(val_pred, np.float32)
                val_ci_value = concordance_index(val_FFS_time, val_pred, val_FFS_event)
                line = 'validation cohort, CI: %.4f, epoch: %d' % (val_ci_value, epoch)
                print(line)
                if (val_ci_value-best_val_C_index)/(best_val_C_index + 1e-8) > 0.005:
                    best_val_C_index = val_ci_value
                    num_up = 0
                elif num_up > 4:
                    #save model
                    checkpoint_name = os.path.join(checkpoint_path, 'model_epoch')
                    saver.save(sess, checkpoint_name, global_step = epoch)
                    break
                else:
                    num_up += 1

if __name__ == '__main__':
    main()