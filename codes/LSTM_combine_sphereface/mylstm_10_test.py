import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
#from tensorflow.examples.tutorials.mnist import input_data
import time
import data
import os
import scipy.io as scio
from Loss_ASoftmax import Loss_ASoftmax
from tensorflow.python import pywrap_tensorflow
import h5py
# tf.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"]='3'
batch_size = tf.placeholder(tf.int32,[])
label_mat=h5py.File('label_zuo.mat')
label_test=label_mat['label'][:]

def getcell(hidden_size):
    lstm_cell = rnn.BasicLSTMCell(
        num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
   # lstm_cell.call(inputs=inputs)
    lstm_cell = rnn.DropoutWrapper(
        cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
    return lstm_cell


with tf.Session() as sess:
   i=19
   _batch_size =4685
   mat_name='data_LBP_differ/data_zuo.mat'
   images, sign = data.test_input(mat_name,_batch_size)
   model_file = './model_LBP_differ/zuo/'
   print("sign:{},images:{}".format(np.shape(sign), np.shape(images)))
   input_size =118  # dim of the input feature
   timestep_size = 9   # each prediction need 28 samples
   hidden_size = 128    # number of hidden layres
   layer_num = 2        # LSTM layer number
   class_num = 1       # class of output .should be 1 if regression

   X = tf.placeholder(tf.float32, [_batch_size, timestep_size, input_size])
   y = tf.placeholder(tf.int64, [_batch_size])

   # define LSTM

   mlstm_cell = rnn.MultiRNNCell([getcell(hidden_size)
                               for _ in range(layer_num)], state_is_tuple=True)

   # use zero to init
   init_state = mlstm_cell.zero_state(_batch_size, dtype=tf.float32)

   # code the train process by myself
   outputs = list()
   state = init_state

   with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
        outputs.append(cell_output)
   #h_state = outputs[-1]
   h_state_1 = outputs[-1]
   h_state_2 = outputs[-1 - 1]
   h_state = tf.concat([h_state_1, h_state_2], 1)
   logits, cross_entropy = Loss_ASoftmax(x=h_state, y=y, l=1.0, num_cls=2, m=4)
   # res=tf.argmax(logits,1)
   # sess.run(tf.global_variables_initializer())
   saver = tf.train.Saver(max_to_keep=100)
   # acc=0.5
   # reader = pywrap_tensorflow.NewCheckpointReader(
#     'model_LBP_differ/zuo/locate/0.797386lstm.ckpt')
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
#     print(reader.get_tensor(key))
   saver.restore(sess,'model_LBP_differ/zuo/locate/0.797386lstm.ckpt')
   logit = sess.run(logits, feed_dict={
                  X: images, y: sign, batch_size: _batch_size})
   print("process1")
   str_='data_temporal_test/res_zuo_0.797386.mat'
   scio.savemat(str_,{'logit':logit,})

   saver.restore(sess,'model_LBP_differ/zuo/locate/0.758170lstm.ckpt')
   logit = sess.run(logits, feed_dict={
                  X: images, y: sign, batch_size: _batch_size})
   print("process2")
   str_='data_temporal_test/res_zuo_0.758170.mat'
   scio.savemat(str_,{'logit':logit,})

   saver.restore(sess,'model_LBP_differ/zuo/locate/0.732026lstm.ckpt')
   logit = sess.run(logits, feed_dict={
                  X: images, y: sign, batch_size: _batch_size})
   print("process3")
   str_='data_temporal_test/res_zuo_0.732026.mat'
   scio.savemat(str_,{'logit':logit,})

   saver.restore(sess,'model_LBP_differ/zuo/locate/10lstm.ckpt-50000')
   logit = sess.run(logits, feed_dict={
                  X: images, y: sign, batch_size: _batch_size})

   print("process4")
   str_='data_temporal_test/res_zuo_15400.mat'
   scio.savemat(str_,{'logit':logit,})
# for 
# while True:
#   ckpt=tf.train.get_checkpoint_state(model_file)
#   if ckpt and ckpt.model_checkpoint_path:
#      saver.restore(sess, ckpt.model_checkpoint_path)
#      #saver.restore(sess,'model_LBP_differ/zuo/10lstm.ckpt-49300')
#      #coord=tf.train.Coordinator()
#      #threads=tf.train.start_queue_runners(sess=sess,coord=coord):
#      #t0=time.time()
#      res_ = sess.run(res, feed_dict={
#                   X: images, y: sign, batch_size: _batch_size})
#      #t1=time.time()
#      #print(t1-t0)
#      count = 0
#      current_pre = 0
#      for sig in res_:
#        if sig==sign[count]:       
#          current_pre += 1
#        count += 1
#      print(res_)  
#      accuracy = float(current_pre) / count
#      print(accuracy)
#      saver.save(sess, 'model_layer4_differ/zuo/%flstm.ckpt' %accuracy)
#      str_='model_layer4_differ/zuo/'+str(accuracy)+'.mat'
#      scio.savemat(str_,{'res_':res_,'sign':sign})
#   time.sleep(2)
