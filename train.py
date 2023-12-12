import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow import ConfigProto

import time, os
import scipy.io as sio
import numpy as np
from random import shuffle

from nets import nets_factory

home_path = os.path.dirname(os.path.abspath(__file__))

tf.app.flags.DEFINE_string('train_dir', home_path + '/test',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('fold', '1',
                           'Directory where checkpoints and event logs are written to.')

FLAGS = tf.app.flags.FLAGS
def main(_):
    ### define path and hyper-parameter
    model_name   = 'STGNN'
    Learning_rate =1e-2

    batch_size = 128
    val_batch_size = 1
    train_epoch =700
    
    weight_decay = 5e-4

    should_log          = 200
    save_summaries_secs = 20
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_num = '0'
        
    train_images, train_labels, val_images, val_labels, train_ed, val_ed = load_data(home_path, FLAGS.fold)
    #val_batch_size = val_images.shape[0]

    dataset_len, *image_size = train_images.shape
    #dataset_len_, *image_size_ed = train_images.shape
    num_label = int(np.max(train_labels)+1)
    with tf.Graph().as_default() as graph:
        # make placeholder for inputs
        image_ph = tf.placeholder(tf.float32, [None]+image_size)
        image_ed_ph = tf.placeholder(tf.float32,[None]+image_size_ed)
        
        label_ph = tf.placeholder(tf.int32, [None])
        is_training_ph = tf.placeholder(tf.int32,[])
        is_training = tf.equal(is_training_ph, 1)
        
        # pre-processing
        image, image_ed = pre_processing(image_ph, image_ed_ph, is_training)
        label = tf.contrib.layers.one_hot_encoding(label_ph, num_label, on_value=1.0)
     
        # make global step
        global_step = tf.train.create_global_step()
        epoch = tf.floor_div(tf.cast(global_step, tf.float32)*batch_size, dataset_len)
        max_number_of_steps = int(dataset_len*train_epoch)//batch_size+1

        # make learning rate scheduler
        LR = learning_rate_scheduler(Learning_rate, [epoch, train_epoch], [0.3, 0.6, 0.8], 0.1)
        
        ## load Net
        class_loss, accuracy, pred = MODEL(model_name, weight_decay, image, image_ed, label, is_training)
        
        #make training operator
        variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        teacher_variables = tf.get_collection('Teacher')
        variables = list(set(variables)-set(teacher_variables))
        
        # make optimizer w/ learning rate scheduler
        optimize = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
        # training main-task
        total_loss = class_loss + tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('loss/total_loss', total_loss)
        gradients  = optimize.compute_gradients(total_loss, var_list = variables)
            
        # merge update operators and make train operator
        update_ops.append(optimize.apply_gradients(gradients, global_step=global_step))
        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')
        
        ## collect summary ops for plotting in tensorboard
        summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES), name='summary_op')
        
        ## make placeholder and summary op for training and validation results
        train_acc_place = tf.placeholder(dtype=tf.float32)
        val_acc_place   = tf.placeholder(dtype=tf.float32)
        val_summary = [tf.summary.scalar('accuracy/training_accuracy',   train_acc_place),
                       tf.summary.scalar('accuracy/validation_accuracy', val_acc_place)]
        val_summary_op = tf.summary.merge(list(val_summary), name='val_summary_op')
        
        ## start training
        train_writer = tf.summary.FileWriter('%s'%FLAGS.train_dir,graph,flush_secs=save_summaries_secs)
        config = ConfigProto()
        config.gpu_options.visible_device_list = gpu_num
        config.gpu_options.allow_growth=True
        
        val_itr = len(val_labels)//val_batch_size
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
          
            sum_train_accuracy = []; time_elapsed = []; total_loss = []
            idx = list(range(train_labels.shape[0]))
            shuffle(idx)
            epoch_ = 0
            for step in range(max_number_of_steps):
                start_time = time.time()
                
                ## feed data

                tl, log, train_acc = sess.run([train_op, summary_op, accuracy],
                                              feed_dict = {image_ph : train_images[idx[:batch_size]],
                                                           image_ed_ph : train_images[idx[:batch_size]],
                                                           label_ph : train_labels[idx[:batch_size]],
                                                           is_training_ph : 1})
    
                time_elapsed.append( time.time() - start_time )
                total_loss.append(tl)
                sum_train_accuracy.append(train_acc)
                idx[:batch_size] = []
                if len(idx) < batch_size:
                    idx_ = list(range(train_labels.shape[0]))
                    shuffle(idx_)
                    idx += idx_
                
                step += 1
                if (step*batch_size)//dataset_len>=epoch_:
                    ## do validation
                    sum_val_accuracy = []
                    val_preds = []
                    for i in range(val_itr):
                        val_batch = val_images[i*val_batch_size:(i+1)*val_batch_size]
                        acc, val_pred = sess.run([accuracy, pred], feed_dict = {image_ph : val_batch,
                                                              image_ed_ph : val_batch,
                                                              label_ph : val_labels[i*val_batch_size:(i+1)*val_batch_size],
                                                              is_training_ph : 0})
                        sum_val_accuracy.append(acc)
                        val_preds.append(val_pred)
                    val_preds = np.hstack(val_preds)    
                    
                    sum_train_accuracy = np.mean(sum_train_accuracy)*100
                    sum_val_accuracy= np.mean(sum_val_accuracy)*100
                    if epoch_%10 == 0:
                        tf.logging.info('Epoch %s Step %s - train_Accuracy : %.2f%%  val_Accuracy : %.2f%%'
                                        %(str(epoch_).rjust(3, '0'), str(step).rjust(6, '0'), 
                                        sum_train_accuracy, sum_val_accuracy))

                    result_log = sess.run(val_summary_op, feed_dict={train_acc_place : sum_train_accuracy,
                                                                     val_acc_place   : sum_val_accuracy   })
                    if step == max_number_of_steps:
                        train_writer.add_summary(result_log, train_epoch)
                    else:
                        train_writer.add_summary(result_log, epoch_)
                    sum_train_accuracy = []

                    epoch_ += 1
                    
                if step % should_log == 0:
                    tf.logging.info('global step %s: loss = %.4f (%.3f sec/step)',str(step).rjust(6, '0'), np.mean(total_loss), np.mean(time_elapsed))
                    train_writer.add_summary(log, step)
                    time_elapsed = []
                    total_loss = []
                
                elif (step*batch_size) % dataset_len == 0:
                    train_writer.add_summary(log, step)

            ## save variables to use for something
            var = {}
            variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)+tf.get_collection('BN_collection')
            for v in variables:
                var[v.name[:-2]] = sess.run(v)
            sio.savemat(FLAGS.train_dir + '/train_params.mat',var)
            sio.savemat(FLAGS.train_dir + '/validation_pred.mat',{'GT' : val_labels, 'Pred' : val_preds})
            
            ## close all
            tf.logging.info('Finished training! Saving model to disk.')
            train_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.STOP))
            train_writer.close()

def MODEL(model_name, weight_decay, image, image_ed, label, is_training):
    network_fn = nets_factory.get_network_fn(model_name, weight_decay = weight_decay)
    end_points = network_fn(image, image_ed, label, is_training=is_training)
    loss = tf.losses.softmax_cross_entropy(label,end_points['Logits'])
    pred = tf.to_int32(tf.argmax(end_points['Logits'], 1))
    accuracy = tf.contrib.metrics.accuracy(pred, tf.to_int32(tf.argmax(label, 1)))
    return loss, accuracy, pred
    
def load_data(home_path, fold_num):
    import scipy as sp
    
    dataset = 'ck_192111'
    joint_pth = 'joint'
    label = 'label'
    bone_pth = 'bone'
    
    with sp.load(home_path + '/%s/%s/train_%s.npz'%(dataset,joint_pth, fold_num)) as f:
        train_ve = [f['arr_%d' % i] for i in range(len(f.files))][0]
    with sp.load(home_path + '/%s/%s/lbtrain_%s.npz'%(dataset,label, fold_num)) as f:
        train_labels = [f['arr_%d' % i] for i in range(len(f.files))][0]-1
        
    with sp.load(home_path + '/%s/%s/test_%s.npz'%(dataset,joint_pth, fold_num)) as f:
        val_ve = [f['arr_%d' % i] for i in range(len(f.files))][0]
    with sp.load(home_path + '/%s/%s/lb_test%s.npz'%(dataset,label, fold_num)) as f:
        val_labels = [f['arr_%d' % i] for i in range(len(f.files))][0]-1

    with sp.load(home_path + '/%s/%s/train_%s.npz'%(dataset,bone_pth, fold_num)) as f:
        train_ed = [f['arr_%d' % i] for i in range(len(f.files))][0]
        
    with sp.load(home_path + '/%s/%s/test_%s.npz'%(dataset,bone_pth, fold_num)) as f:
        val_ed = [f['arr_%d' % i] for i in range(len(f.files))][0]  

    return train_ve, train_labels[:,0], val_ve, val_labels[:,0], train_ed, val_ed

def pre_processing(image, image_ed, is_training):
    with tf.variable_scope('preprocessing'):
        return image, image_ed

def learning_rate_scheduler(Learning_rate, epochs, decay_point, decay_rate):
    with tf.variable_scope('learning_rate_scheduler'):
        e, te = epochs
        for i, dp in enumerate(decay_point):
            Learning_rate = tf.cond(tf.greater_equal(e, int(te*dp)), lambda : Learning_rate*decay_rate, 
                                                                          lambda : Learning_rate)
        tf.summary.scalar('learning_rate', Learning_rate)
        return Learning_rate

if __name__ == '__main__':
    tf.app.run()

