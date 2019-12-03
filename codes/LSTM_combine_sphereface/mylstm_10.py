import logging
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
#from tensorflow.examples.tutorials.mnist import input_data
import data
import os
import scipy.io as scio
from Loss_ASoftmax import Loss_ASoftmax
from tensorflow.python import pywrap_tensorflow
# tf.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def next_batch(images, sign, num):
    data1 = images
    data2 = sign
    # print data1.shape
    # print data2.shape
    idx = np.arange(0, len(data1))
    np.random.shuffle(idx)
    idx = idx[0:num]
    data_shuffle1 = [data1[i] for i in idx]
    data_shuffle1 = np.asarray(data_shuffle1)
    data_shuffle2 = [int(data2[i]) for i in idx]
    # print data_shuffle2
    data_shuffle2 = np.asarray(data_shuffle2)
    # print data_shuffle2
    # data_shuffle=(data_shuffle1,data_shuffle2)
    # print data_shuffle1.shape
    # print data_shuffle2.shape
    return data_shuffle1, data_shuffle2, idx


error = [0 for i in range(1500)]

sess = tf.Session()
#mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
# print mnist.train.images.shape
images, sign = data.data_input('data_LBP_differ/train_zuo.mat')
logger.info('images.shape:')
logger.info(images.shape)
#print ("images.shape:%s" % images.shape)
#print labels.shape
# print sign.shape

# lr = 5e-4
learning_rate=tf.placeholder(tf.float32,shape=None)
def ache_rate(step):
  if step<100:
    lr=0.01
  elif step<3000:
    lr=0.001
  elif step<30000:
    lr=0.0001
  elif step<60000:
    lr=0.00001
  else:
    lr=0.000001
  return lr
batch_size = tf.placeholder(tf.int32, [])
input_size = 118  # dim of the input feature
timestep_size = 9   # each prediction need 28 samples
hidden_size = 128    # number of hidden layres
layer_num = 2       # LSTM layer number
class_num = 1       # class of output .should be 1 if regression

_batch_size = 4
X = tf.placeholder(tf.float32, [_batch_size, timestep_size, input_size])
y = tf.placeholder(tf.int64, [_batch_size])
# diffrent batchsize when train and test so use place holder
# batch_size = tf.placeholder(tf.int32, [])  # the type must be tf.int32, batch_size = 128
keep_prob = tf.placeholder(tf.float32, [])


# define input of lstm
# X=tf.reshape(_X,[-1,28,28])


# define LSTM
def getcell(hidden_size, keep_prob):
    lstm_cell = rnn.BasicLSTMCell(
        num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
# lstm_cell.call(inputs=inputs)
    lstm_cell = rnn.DropoutWrapper(
        cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell


mlstm_cell = rnn.MultiRNNCell([getcell(hidden_size, keep_prob)
                               for _ in range(layer_num)], state_is_tuple=True)

# use zero to init
init_state = mlstm_cell.zero_state(_batch_size, dtype=tf.float32)

#outputs.shape = [batch_size, timestep_size, hidden_size]
#state.shape = [layer_num, 2, batch_size, hidden_size]
# code the train process by myself
outputs = list()
state = init_state

with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
        outputs.append(cell_output)
h_state = outputs[-1]
# print h_state
h_state_1 = outputs[-1]
h_state_2 = outputs[-1 - 1]
h_state = tf.concat([h_state_1, h_state_2], 1)
print("hstate:%s" % h_state)
# cross_entropy=-tf.reduce_mean(y*tf.log(y_pre))+0.005*tf.nn.l2_loss(W)
logits, cross_entropy = Loss_ASoftmax(x=h_state, y=y, l=1.0, num_cls=2, m=4)
train_op = tf.train.AdamOptimizer(learning_rate,beta1=0.5,beta2=0.9).minimize(
    cross_entropy)
res=tf.argmax(logits,1)
saver = tf.train.Saver(max_to_keep=10)
sess.run(tf.global_variables_initializer())
j = 0
for i in range(50000):
    lrn_rate=ache_rate(i)
    # images=np.reshape(batch[1],[batchsize,118,1])
    # print "step %d, training eeror %g" % ( (i+1), train_accuracy)
    train_x, train_y, idx = next_batch(images, sign, _batch_size)
    # print batch[0].shape
    if (i + 1) % 100 == 0:
        train_error = sess.run(cross_entropy, feed_dict={
            X: train_x, y: train_y, keep_prob: 1.0, batch_size: _batch_size,learning_rate:lrn_rate})
        # epoch number: mnist.train.:epochs_complete
        error[j] = train_error
        j = j + 1
        res_ = sess.run(res, feed_dict={
            X: train_x, y: train_y, keep_prob: 1.0, batch_size: _batch_size,learning_rate:lrn_rate})
        saver.save(sess, 'model_LBP_differ/zuo/10lstm.ckpt',global_step=i+1)
        logger.info("step %d, training error %g,learning_rate:%g" %
                    ((i + 1), train_error, lrn_rate))
        print "res:",res_
        print "true:",train_y
    sess.run(train_op, feed_dict={
             X: train_x, y: train_y, keep_prob: 0.5, batch_size: _batch_size,learning_rate:lrn_rate})
"""reader = pywrap_tensorflow.NewCheckpointReader(
    'model_movie/model2/10lstm.ckpt-4000')
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))"""
# accuracy
#logger.info("error:")
#logger.info(error)
#scio.savemat('result/error10.mat', {'error': error})
