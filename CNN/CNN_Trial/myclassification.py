from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
from numpy import array
import cv2
pickle_file = 'anirudh.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']

  del save  # hint to help gc free up memory
  #print('Training set', train_dataset.shape, train_labels.shape)
  #print('Validation set', valid_dataset.shape, valid_labels.shape)
  #print('Test set', test_dataset.shape, test_labels.shape)
  image_size = 28
num_labels = 3
num_channels = 1 # grayscale
def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size*image_size)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)

print('Training set', train_dataset.shape, train_labels.shape)




n_classes = 3
batch_size=20

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    train_prediction=tf.nn.softmax(prediction)
    
    num_steps = 501
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Initialized')

        for step in range(num_steps):
            offset=(step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_data, y: batch_labels})
            if(step%50==0):
                print('Minibatch loss at step %d: %f' % (step, c))
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:train_dataset, y:train_labels}))
        a=cv2.imread('a.png')
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        a= cv2.normalize(a.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        a=array(a)
        a=a.reshape(1,784)
        print(sess.run(accuracy, feed_dict={x: a,y: [[1.0, 0.0, 0.0]]}))
        print('probablity:',train_prediction.eval({x:a,y:[[1.0, 0.0, 0.0]]}))
        
        
train_neural_network(x)