
import os
import tensorflow as tf
import numpy as np
import scipy.io
from datetime import datetime
import pandas as pd
from lifelines.utils import concordance_index

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


momentum = 0.95
learning_rate_decay = 0.95
learning_rate_base = 1e-4
learning_rate_step = 500000
warmup_step = 80

alpha = 1.5
belta = 1.0
gamma = 1.0
sita = 1.0
num_event = 4
reg_factor = 5.0
intra_loss_weight = [1.0, 0.1]

sigma1 = 1.0
dim_interact_feature =512

snapshot_dir = "/YourOutputPath2/64_multi-sequence_MR+clinic_3/checkpoint_path/model_epoch-23"
output_path = "/YourOutputPath2/"

feat_path1 = "/YourOutputPath1/T1/T1_dlFeature.csv"
feat_path2 ="/YourOutputPath1/T1C/T1C_dlFeature.csv"
feat_path3 = "/YourOutputPath1/T2/T2_dlFeature.csv"
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    DL_feat1 = pd.read_csv(feat_path1, header = 0, index_col = 0)
    DL_feat2 = pd.read_csv(feat_path2, header = 0, index_col = 0)
    DL_feat3 = pd.read_csv(feat_path3, header = 0, index_col = 0)
    tra_No_ind = np.array(range(len(DL_feat1)), np.int32)
    tra_feat1 = DL_feat1.iloc[tra_No_ind,:]
    tra_feat2 = DL_feat2.iloc[tra_No_ind,:]
    tra_feat3 = DL_feat3.iloc[tra_No_ind,:]
    tra_feat = pd.concat([tra_feat1,tra_feat2,tra_feat3],axis=1,ignore_index=True)
    feature_num = tra_feat.shape[1]
    tra_Pat_ID = np.array(DL_feat1.index)

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

def _create_fc_layer(x, output_dim, activation, scope, keep_prob = None,use_bias=True, w_reg = None, initial_W = None):
    if initial_W is None:
        initial_W = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(scope):
        layer_out = tf.layers.dense(inputs=x, use_bias=use_bias, units=output_dim, kernel_initializer=initial_W, kernel_regularizer=w_reg)
        if keep_prob is not None:
            layer_out = tf.nn.dropout(layer_out, keep_prob=keep_prob)
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
    DMFS_loss_cox = DeepSurv_loss(s_time[:,1], s_event[:,1], Pat_ind[:,1], output[:,1])
    LRFS_loss_cox = DeepSurv_loss(s_time[:,2], s_event[:,2], Pat_ind[:,2], output[:,2])

    DFS_loss_cox = DeepSurv_loss(s_time[:,3], s_event[:,3], Pat_ind[:,3], tf.reduce_max(output,1))

    loss_cox = alpha*DFS_loss_cox + belta*OS_loss_cox + gamma*DMFS_loss_cox + sita*LRFS_loss_cox
    return loss_cox

def main():
    with tf.device('/gpu:0'):
        x = tf.placeholder(tf.float32, [None, clinic_num], name = 'input')
        s_time = tf.placeholder(tf.float32, [None,num_event], name = 'surv_time')
        s_event = tf.placeholder(tf.float32, [None,num_event], name = 'surv_event')
        Pat_ind = tf.placeholder(tf.int32, [None,num_event], name = 'Pat_ind')
        keep_prob = tf.placeholder(tf.float32, name = 'keep_rate')
        treatment = tf.placeholder(tf.float32, [None, dim_interact_feature], name = 'treatment')
        global_step = tf.placeholder(tf.int32, [])
        # model
        fc = _create_fc_layer(x, 3*clinic_num, 'relu', 'shared_layer1', keep_prob, w_reg = reg_W)

        fc1_1 = _create_fc_layer(fc, clinic_num, 'relu', 'specific_layer1_1', keep_prob, w_reg = reg_W)
        fc1_2 = _create_fc_layer(fc1_1, dim_interact_feature, 'relu', 'specific_layer1_2', keep_prob, w_reg = reg_W)
        output1 = _create_fc_layer(fc1_2, num_event-1, 'tanh', 'output_1', use_bias= False, w_reg = reg_W_out)

        fc2_1 = _create_fc_layer(fc, clinic_num, 'relu', 'specific_layer2_1', keep_prob, w_reg = tf.contrib.layers.l2_regularizer(scale=0.005))
        fc2_2 = _create_fc_layer(fc2_1, dim_interact_feature, 'relu', 'specific_layer2_2', keep_prob, w_reg = tf.contrib.layers.l2_regularizer(scale=0.005))
        fc2_2 = tf.multiply(treatment, fc2_2)
        output2 = _create_fc_layer(fc2_2, 1, 'tanh', 'output_2', use_bias= False, w_reg = tf.contrib.layers.l1_regularizer(scale=0.05))
        # loss
        loss_cox_prog = Get_loss(output1, s_time, s_event, Pat_ind)
        pred_DFS =  tf.py_func(np.max, [output1,1], tf.float32)
        loss_cox_pred = DeepSurv_loss(s_time[:,3], s_event[:,3], Pat_ind[:,3], output2)
        loss_reg = tf.losses.get_regularization_loss()

        loss_total = intra_loss_weight[0]*loss_cox_prog + intra_loss_weight[1]*loss_cox_pred + reg_factor*loss_reg

        learning_rate = exponential_decay_with_warmup(warmup_step,learning_rate_base,global_step,learning_rate_step,learning_rate_decay,staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = momentum, use_nesterov = True)

        train_step = optimizer.minimize(loss_total)

        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)#设置每个GPU使用率0.7代表70%
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, snapshot_dir)
            DL_feature = pd.DataFrame()
            for i in range(len(tra_Pat_ID)):
                treat = np.array([1]*dim_interact_feature)
                Pat_pred, pred_response = sess.run([pred_DFS, output2], feed_dict = {x: tra_feat[i,:].reshape(1,clinic_num), keep_prob: 1.0, treatment: treat.reshape(1,dim_interact_feature)}) 
                preds = np.append(Pat_pred, pred_response)
                aFeature = pd.Series(preds)
                aFeature = aFeature.to_frame()
                aFeature.columns = [tra_Pat_ID[i]]
                DL_feature = DL_feature.append(aFeature.T)

            # import pdb;pdb.set_trace()
            wb = os.path.join(output_path, 'MR_RadSigs.csv')
            DL_feature.columns = ['Pat_ID', 'Prog_score', 'Pred_score']
            print(wb)
            DL_feature.to_csv(path_or_buf = wb, encoding='utf-8', index = True, header = True)

if __name__ == '__main__':
    main()