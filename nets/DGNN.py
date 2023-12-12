from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as tcl
from nets import directed_graph as grp


def DGNN_arg_scope(weight_decay=0.0005):
    with tf.contrib.framework.arg_scope([tcl.conv2d, tcl.fully_connected], trainable = True,
                                        weights_initializer=tcl.variance_scaling_initializer(mode='FAN_OUT'),
                                        weights_regularizer=tcl.l2_regularizer(weight_decay),
                                        biases_initializer=tf.zeros_initializer(), activation_fn = None):
        with tf.contrib.framework.arg_scope([tcl.batch_norm], trainable = True,
                                            scale = True, center = True, activation_fn=None, decay=0.9, epsilon = 1e-5,
                                            param_regularizers={'gamma': tcl.l2_regularizer(weight_decay),
                                                                'beta' : tcl.l2_regularizer(weight_decay)},
                                            variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'BN_collection']) as arg_sc:
            return arg_sc

def temporal_block(x, depth, Kt, stride, is_training=False, name=" "):
    
    with tf.variable_scope(name):
        _,T,n,_ = x.get_shape().as_list()
        T_=int(T/stride)
        x_input = x
        x_input = x_input[:,:T_,:,:]
        
        wt = tf.get_variable(name='wt', shape=[Kt, 1, depth, depth], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([depth]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, [stride, 1], padding='SAME') + bt

    return (x_conv[:, :, :, 0:depth] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -depth:])

  
def BiTemporalConv(x_ve, x_ed, depth, kn_size=9, stride=1, is_training=False, name=' '):
    
        with tf.variable_scope(name):
            x_ve = tcl.conv2d(x_ve, depth, [kn_size, 1], [stride, 1],  scope='Conv2d_1') #padding=[[int((kn_size-1)/2),0],[0,0],[0,0],[0,0]])
            x_ve = tcl.batch_norm(x_ve, scope='btc_1', activation_fn = None, trainable = True, is_training = is_training)
            x_ed = tcl.conv2d(x_ed, depth, [kn_size, 1], [stride, 1],  scope='Conv2d_2')
            x_ed = tcl.batch_norm(x_ed,scope='btc_2', activation_fn = None, trainable = True, is_training = is_training)
        
        return x_ve, x_ed

def DGNBlock(x_ve, x_ed, depth, source_m, target_m, is_training=False, name=' '):

     with tf.variable_scope(name):
         fv = x_ve
         fe = x_ed
         source_M = tf.get_variable('source_M', list(source_m.shape), tf.float32, trainable = True,
                                             initializer=tf.ones_initializer())
         source_M *= source_m
         target_M = tf.get_variable('target_M', list(source_m.shape), tf.float32, trainable = True,
                                             initializer=tf.ones_initializer())
         target_M *= source_m    
         fv = tf.transpose(fv,[0, 3, 1, 2])
         fe = tf.transpose(fe,[0, 3, 1, 2])
         N, C, T, V_node = fv.get_shape().as_list()
         _, _, _, V_edge = fe.get_shape().as_list()
         # Reshape for matmul, shape: (N, CT, V)
         fv = tf.transpose(fv,[0,1,3,2])
         fv = tf.reshape(fv,[-1,C * T, V_node])
    
         fe = tf.transpose(fe,[0,1,3,2])
         fe = tf.reshape(fe,[-1,C * T, V_edge])     
         # Compute features for node/edge updates
         fe_in_agg = tf.einsum('nce,ev->ncv', fe, source_M)
         fe_out_agg = tf.einsum('nce,ev->ncv', fe, target_M)
         fvp = tf.stack([fv, fe_in_agg, fe_out_agg],axis=1)   # Out shape: (N,3,CT,V_nodes)
         fvp = tf.reshape(fvp,[-1,T,V_node,3*C])
         fvp = tcl.fully_connected(fvp,depth, scope='fvp_fc')#.permute(0,3,1,2)    # (N,C_out,T,V_node)
         fvp = tf.reshape(fvp,[-1,T,V_node,depth])
         fvp = tcl.batch_norm(fvp, scope='fvp_bc', activation_fn = None, trainable = True, is_training = is_training)
         fvp = tf.nn.relu(fvp)
         
         fv_in_agg = tf.einsum('ncv,ve->nce', fv, source_M)
         fv_out_agg = tf.einsum('ncv,ve->nce', fv, target_M)
         fep = tf.stack([fe, fv_in_agg, fv_out_agg], axis=1)   # Out shape: (N,3,CT,V_edges)
         fep = tf.reshape(fep,[-1,T,V_edge,3*C])
         fep = tcl.fully_connected(fep,depth, scope='fep_fe')#.permute(0,3,1,2)    # (N,C_out,T,V_edge)
         fep = tf.reshape(fep,[-1,T,V_edge,depth])
         fep = tcl.batch_norm(fep, scope='fep_bc', activation_fn = None, trainable = True, is_training = is_training)
         fep = tf.nn.relu(fep)

     return fvp, fep

def GraphTemporalConv(x_ve, x_ed, depth, source_m, target_m, kn_size=9, stride=1, is_training= False,residual=True,name=' '):
    
    with tf.variable_scope(name):
        fv_res, fe_res = BiTemporalConv(x_ve, x_ed, depth, kn_size, stride, is_training=is_training, name='Biotemporoal_block1')
        
        fv, fe = DGNBlock(x_ve, x_ed, depth, source_m, target_m, is_training=is_training, name='DGN_block')
        fv = temporal_block(fv, depth, 3, stride, is_training=is_training, name="temp_blk_fv1")
        fe = temporal_block(fe, depth, 3, stride, is_training=is_training, name="temp_blk_fe1")
        
        fv += fv_res
        fe += fe_res
        
    return tf.nn.relu(fv), tf.nn.relu(fe)

def data_normalization(x, is_training=False, name=' ', scope=' '):
    
    with tf.variable_scope('pre_process'):
        
        
        x = tf.transpose(x,[0, 4, 2, 3, 1])
        x = tf.slice(x,[0,0,0,0,0],[-1,-1,-1,-1,2]) # positoin
 
        x -= tf.slice(x, [0,0,0,13,0], [-1,-1,-1,1,-1])
        _, var = tf.nn.moments(x,3,keepdims=True)
        x /= tf.sqrt(var+1e-8)
        
        N,M,T,V,C = x.get_shape().as_list() 

        x = tf.reshape(x,[-1,M*V*C,T])
        x = tcl.batch_norm(x, scope=scope, activation_fn = None, trainable = True, is_training = is_training )
        x = tf.reshape(x,[-1,M,V,C,T])
        x = tf.transpose(x,[0,1,3,4,2])
        x = tf.reshape(x,[-1, T, V, C])
            
        return x

def DGNN(fe, fv, label, is_training):
    end_points = {}
    
    fe = data_normalization(fe, is_training=True, name='fe_fc', scope='bc1' )
    fv = data_normalization(fv, is_training=True, name='fv_fc', scope='bc2' )
        
    source_m, target_m = grp.Graph(51)
        
    with tf.variable_scope('STGNN'):
        
        fv, fe = GraphTemporalConv(fv, fe, 8, source_m, target_m, 9,2, is_training = is_training, residual=False, name='Grp2')
        fv, fe = GraphTemporalConv(fv, fe, 16, source_m, target_m, 9,2, is_training = is_training, residual=False, name='Grp4')
        fv, fe = GraphTemporalConv(fv, fe, 16, source_m, target_m, 9,1, is_training = is_training, residual=False, name='Grp5')
        fv, fe = GraphTemporalConv(fv, fe, 32, source_m, target_m, 9,2, is_training = is_training, residual=False, name='Grp6')
        fv, fe = GraphTemporalConv(fv, fe, 32, source_m, target_m, 9,1, is_training = is_training, residual=False, name='Grp7')
        fv, fe = GraphTemporalConv(fv, fe, 64, source_m, target_m, 9,2, is_training = is_training, residual=False, name='Grp8')
        fv, fe = GraphTemporalConv(fv, fe, 64, source_m, target_m, 9,1, is_training = is_training, residual=False, name='Grp9')
        
        fv = tf.reduce_mean(fv, [1,2])
        fe = tf.reduce_mean(fe, [1,2])
        
        out = tf.concat([fv, fe], axis=-1)


        logits = tcl.fully_connected(out, label.get_shape().as_list()[-1],
                                     biases_initializer = tf.zeros_initializer(),
                                     biases_regularizer = tf.contrib.layers.l2_regularizer(5e-4),
                                     trainable=True, scope = 'full')
        end_points['Logits'] = logits
    
    return end_points


















