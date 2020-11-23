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

#Fixed hyper-parameters
dim_interact_feature = 512
momentum = 0.95
num_event = 4
sigma1 = 1.0
num_epochs = 100


######
#The hyper-parameters were determined using Dataset D3
######
learning_rate_decay = 0.95
learning_rate_base = 1e-4
learning_rate_step = 500000 #Training model using fixed learning rate
warmup_step = 50

batch_size = 64
keep_prob_rate = 0.6
reg_W = tf.contrib.layers.l2_regularizer(scale=5e-2) #weight regularization for fully connected layers
reg_W_out = tf.contrib.layers.l1_regularizer(scale=1e-2) #weight regularization for output layers

#weights of different losses
alpha = 1.5
belta = 1.0
gamma = 1.0
sita = 1.0
reg_factor = 5.0
intra_loss_weight = [1.0, 1.0]


#set No. of patients with event : No. of censored patients as 0.7
sampling_rate = 0.7 #Sampling ratio

output_path = "/YourOutputPath2"
if not os.path.exists(output_path):
    os.mkdir(output_path)
output_path = os.path.join(output_path, '64_multi-sequence_MR')
checkpoint_path = os.path.join(output_path, "checkpoint_path")
if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
log_path = os.path.join(output_path,'logs.txt')
clinic_path = "matched_data_from_center1.csv"
feat_path1 = "/YourOutputPath1/T1/T1_dlFeature.csv"
feat_path2 = "/YourOutputPath1/T1C/T1C_dlFeature.csv"
feat_path3 = "/YourOutputPath1/T2/T2_dlFeature.csv"
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    DL_feat1 = pd.read_csv(feat_path1, header = 0, index_col = 0)
    DL_feat2 = pd.read_csv(feat_path2, header = 0, index_col = 0)
    DL_feat3 = pd.read_csv(feat_path3, header = 0, index_col = 0)
    clinic_msag = pd.read_csv(clinic_path, header = 0, index_col = 0)

    Pat_ID = DL_feat1.index
    Pat_ID = [str(s) for s in Pat_ID]
    DL_feat1.index = Pat_ID
    DL_feat2.index = Pat_ID
    DL_feat3.index = Pat_ID
    Pat_ID = clinic_msag.index
    Pat_ID = [str(s) for s in Pat_ID]
    #Dataset D1
    DL_feat1 = DL_feat1.loc[Pat_ID]
    #Dataset D3
    DL_feat2 = DL_feat2.loc[Pat_ID]
    #Internal test cohort
    DL_feat3 = DL_feat3.loc[Pat_ID]
    Pat_No_ind = np.array(range(len(DL_feat1)), np.int32)

    tra_No_ind = Pat_No_ind[clinic_msag['data_cohort']==1]
    val_No_ind = Pat_No_ind[clinic_msag['data_cohort1']==1]
    test_No_ind = Pat_No_ind[clinic_msag['data_cohort2']==1]
    tra_msag = clinic_msag.iloc[tra_No_ind,:]
    val_msag = clinic_msag.iloc[val_No_ind,:]
    test_msag = clinic_msag.iloc[test_No_ind,:]

    tra_feat1 = DL_feat1.iloc[tra_No_ind,:]
    tra_feat2 = DL_feat2.iloc[tra_No_ind,:]
    tra_feat3 = DL_feat3.iloc[tra_No_ind,:]
    tra_feat = pd.concat([tra_feat1,tra_feat2,tra_feat3],axis=1,ignore_index=True)

    tra_feat1 = DL_feat1.iloc[val_No_ind,:]
    tra_feat2 = DL_feat2.iloc[val_No_ind,:]
    tra_feat3 = DL_feat3.iloc[val_No_ind,:]
    val_feat = pd.concat([tra_feat1,tra_feat2,tra_feat3],axis=1,ignore_index=True)

    tra_feat1 = DL_feat1.iloc[test_No_ind,:]
    tra_feat2 = DL_feat2.iloc[test_No_ind,:]
    tra_feat3 = DL_feat3.iloc[test_No_ind,:]
    test_feat = pd.concat([tra_feat1,tra_feat2,tra_feat3],axis=1,ignore_index=True)

    tra_Pat_ID = np.array(tra_msag.index)
    tra_treat = np.array(tra_msag.loc[:, 'treatment'], np.float32)
    tra_time = np.array(tra_msag.loc[:, ['OS.time', 'DMFS.time', 'LRRFS.time','DFS.time']], np.float32)
    tra_event = np.array(tra_msag.loc[:, ['OS', 'DMFS', 'LRRFS','DFS']], np.float32)
    tra_FFS_time = np.array(tra_msag.loc[:, 'DFS.time'], np.float32)
    tra_FFS_event = np.array(tra_msag.loc[:, 'DFS'], np.float32)
    tra_FFS_event[tra_FFS_event<0.0] = 0.0

    deep_features = np.array(tra_feat, np.float32)
    feature_num = deep_features.shape[1]

    # import pdb; pdb.set_trace()
    val_Pat_ID = np.array(val_msag.index)
    val_treat = np.array(val_msag.loc[:, 'treatment'], np.float32)
    val_FFS_time = np.array(val_msag.loc[:, 'DFS.time'], np.float32)
    val_FFS_event = np.array(val_msag.loc[:, 'DFS'], np.float32)
    val_FFS_event[val_FFS_event<0.0] = 0.0
    deep_features_val = np.array(val_feat, np.float32)

    test_Pat_ID = np.array(test_msag.index)
    test_treat = np.array(test_msag.loc[:, 'treatment'], np.float32)
    test_FFS_time = np.array(test_msag.loc[:, 'DFS.time'], np.float32)
    test_FFS_event = np.array(test_msag.loc[:, 'DFS'], np.float32)
    test_FFS_event[test_FFS_event<0.0] = 0.0
    deep_features_test = np.array(test_feat, np.float32)

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
    
    input_x1 = deep_features[ind, :]
    
    treat = 0.5*tra_treat[ind]
    treat = treat.reshape((-1, 1))
    treat_out = np.ones((1, dim_interact_feature), np.float32)
    treat_out = treat * treat_out
    
    Input_Pat_ID = tra_Pat_ID[ind]

    
    return treat_out, input_x1, input_time, input_event, input_idx, Input_Pat_ID
    


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
        x = tf.placeholder(tf.float32, [None, feature_num], name = 'input')
        s_time = tf.placeholder(tf.float32, [None,num_event], name = 'surv_time')
        s_event = tf.placeholder(tf.float32, [None,num_event], name = 'surv_event')
        Pat_ind = tf.placeholder(tf.int32, [None,num_event], name = 'Pat_ind')
        keep_prob = tf.placeholder(tf.float32, name = 'keep_rate')
        treatment = tf.placeholder(tf.float32, [None, dim_interact_feature], name = 'treatment')
        global_step = tf.placeholder(tf.int32, [])
        # model for Prognostic-score and Predictive-score
        # shared backbone network
        fc = _create_fc_layer(x, 3*feature_num, 'relu', 'shared_layer1', keep_prob, w_reg = reg_W)
        
        # the subnetwork for predicting prognosis (Prognostic-score)
        fc1_1 = _create_fc_layer(fc, feature_num, 'relu', 'specific_layer1_1', keep_prob, w_reg = tf.contrib.layers.l2_regularizer(scale=0.2))
        fc1_2 = _create_fc_layer(fc1_1, dim_interact_feature, 'relu', 'specific_layer1_2', keep_prob, w_reg = tf.contrib.layers.l2_regularizer(scale=0.2))
        output1 = _create_fc_layer(fc1_2, num_event-1, 'tanh', 'output_1', use_bias= False, w_reg = reg_W_out)
        
        # the subnetwork for predicting treatment response (Predictive-score)
        fc2_1 = _create_fc_layer(fc, feature_num, 'relu', 'specific_layer2_1', keep_prob, w_reg = tf.contrib.layers.l2_regularizer(scale=2e-2))
        fc2_2 = _create_fc_layer(fc2_1, dim_interact_feature, 'relu', 'specific_layer2_2', keep_prob, w_reg = tf.contrib.layers.l2_regularizer(scale=2e-2))
        fc2_2 = tf.multiply(treatment, fc2_2)
        output2 = _create_fc_layer(fc2_2, 1, 'tanh', 'output_2', use_bias= False, w_reg = reg_W_out)

        restore_var = [v for v in tf.trainable_variables()]
        # print(restore_var)
        out_vars = [v for v in restore_var if 'specific_layer1' not in v.name]
        out_vars = [v for v in out_vars if 'output_1' not in v.name]
        # print(out_vars)
        # loss
        loss_cox_prog = Get_loss(output1, s_time, s_event, Pat_ind)
        pred_DFS = tf.py_func(np.max, [output1,1], tf.float32)
        loss_cox_pred = DeepSurv_loss(s_time[:,3], s_event[:,3], Pat_ind[:,3], output2)
        mean_0_loss = tf.abs(tf.reduce_mean(output2)) #make the relative risk of disease progression 0 as possible as
        loss_reg = tf.losses.get_regularization_loss()

        loss_total = intra_loss_weight[0]*loss_cox_prog + intra_loss_weight[1]*loss_cox_pred + reg_factor*loss_reg+0.1*mean_0_loss

        learning_rate = exponential_decay_with_warmup(warmup_step,learning_rate_base,global_step,learning_rate_step,learning_rate_decay,staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = momentum, use_nesterov = True)
        train_step = optimizer.minimize(loss_total,var_list=restore_var)


        saver = tf.train.Saver(max_to_keep = 50)
        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # loader = tf.train.Saver(var_list=restore_var)
            # loader.restore(sess, snapshot_dir)
            
            gsp = 0
            best_val_C_index = 0.0
            results = []
            # Loop over number of epochs
            for epoch in range(num_epochs):
                np.random.shuffle(ind_0)
                np.random.shuffle(ind_1)
                # Initialize iterator with the training dataset
                train_risk = 0.0
                prog_risk = 0.0
                pred_risk = 0.0
                reg_risk = 0.0
                # import pdb;pdb.set_trace()
                for i in range(num_batchs):
                    gsp += 1
                    ind0 = ind_0[i*r0:(i+1)*r0]
                    ind1 = ind_1[i*r1:(i+1)*r1]
                    treat, input_x1, input_time, input_event, input_idx, Input_Pat_ID = GetData(ind0,ind1)
                    v1, _, reg_ls, prog_ls, pred_ls, total_ls, now_lr = sess.run([output2, train_step, mean_0_loss, loss_cox_prog, loss_cox_pred, loss_total, learning_rate], feed_dict = {global_step:gsp, treatment: treat, x: input_x1, s_time: input_time, s_event: input_event, Pat_ind: input_idx, keep_prob: keep_prob_rate})
                    reg_risk += reg_ls
                    train_risk += total_ls
                    prog_risk += prog_ls
                    pred_risk += pred_ls
                print(np.sum(v1))
                reg_risk /= num_batchs
                train_risk /= num_batchs
                prog_risk /= num_batchs
                pred_risk /= num_batchs
                line = 'epoch: %d, learning rate: %.5f, tatol_loss: %.4f, reg_loss: %.4f, prognosis-cox loss: %.4f, predict-cox loss: %.4f' % (epoch + 1, now_lr, train_risk, reg_risk, prog_risk, pred_risk)
                print(line)
                with open(log_path, 'a') as f:
                    f.write(line + '\n')
                
                   
                val_pred = []
                for i in range(len(val_treat)):
                    xd = val_treat[i]
                    treat = np.array([xd]*dim_interact_feature)
                    
                    Pat_pred = sess.run(pred_DFS, feed_dict = {x: deep_features_val[i,:].reshape(1,clinic_num), 
                                                             keep_prob: 1.0, treatment: treat.reshape(1,dim_interact_feature)}) 
                    val_pred.append(-np.exp(Pat_pred))
                
                val_pred = np.array(val_pred, np.float32)
                val_ci_value = concordance_index(val_FFS_time, val_pred, val_FFS_event)
                results.append(val_ci_value)
                line = 'validation cohort, CI: %.4f, epoch: %d' % (val_ci_value, epoch)
                
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
                    
            np.save('/YourOutputPath2/Training_results.npy', np.array(results, np.float32))
                    
                    

if __name__ == '__main__':
    main()