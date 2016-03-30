# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def read_data():
	data = pd.read_csv("/home/avalon/ml/dr/data/train.csv")
	print("data({0[0]},{0[1]})".format(data.shape))
#	print(data.head())
	labels_flat = data[[0]].values
	print("labels_flat({0})".format(len(labels_flat)))
	return data
def dense_to_one_hot(labels_dense, num_classes=10):
	num_labels = labels_dense.shape[0]
	labels_one_hot = np.zeros((num_labels, num_classes))
	'''
	index_offset = [0, 10, 20, 30 ... (num_labels-1) * 10]
	'''
	index_offset = np.arange(num_labels)*num_classes
	
	'''	
	a = [1 2 3]
	b = [1 2 3 4 5 6 7]
	b[a] = 3 
	==>	b = [1 3 3 3 5 6 7]
	G'''

	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

def display(img):
	figure = plt.figure()
	one_image = np.multiply(img.reshape(28,28), 255).astype(np.uint8)
	plt.axis('off')
	plt.imshow(one_image, cmap=cm.binary)

if __name__ == '__main__':
	data = read_data()
	labels_flat = data['label'].values
	labels = dense_to_one_hot(labels_flat)
	labels = labels.astype(np.uint8)
	print("labels({0[0]}, {0[1]})".format(labels.shape))

	images = data.iloc[:, 1:].values
	images = images.astype(np.float)

	images = images * (1.0 / 255.0)
	print("images({0[0]}, {0[1]})".format(images.shape))

	# display image
	display(images[2])
	print(labels_flat[2])
 

# <codecell>

#settings
LEARNING_RATE = 0.01
TRAINING_ITERATIONS = 2000
DISPLAY_STEP = 100
METHOD_NAME = 'softmax'
DROPOUT = 0.5
BATCH_SIZE = 50

RUN_DEEPNET = True

if RUN_DEEPNET:
    LEARNING_RATE = 1e-4
    TRAINING_ITERATIONS = 20000
    DISPLAY_STEP = 1000
    METHOD_NAME = 'deepnet'
    
# set to 0 to train on a whold dataset
VALIDATION_SIZE = 4200

# <codecell>

# split data into validation & training

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

print("train_images({0[0]}, {0[1]})".format(train_images.shape))
print("validation_images({0[0]}, {0[1]})".format(validation_images.shape))

# <codecell>

# read test data from CSV file
test_images = pd.read_csv("../data/test.csv").values
test_images = test_images.astype(np.float)

# convert from [0, 255] -> [0.0, 1.0]
test_images = test_images * (1.0/255.0)
display(test_images[1])
print("test_images({0[0]},{0[1]})".format(test_images.shape))

# <codecell>

import tensorflow as tf

# weight initiallization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution and pooling

# <codecell>

# convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# <codecell>

# setup TensorFlow graph

# input & output
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# readout layer for deep net
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


# readout layer for regression
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


if RUN_DEEPNET:
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
else:
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# optimisation function
if RUN_DEEPNET:
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
else:
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    
# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# prediction function
predict = tf.argmax(y,1)

# <codecell>

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):
    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# <codecell>

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(TRAINING_ITERATIONS):
        
        #get new batch
        batch_xs, batch_ys = next_batch(BATCH_SIZE)        
        
        # check progress
        if i%DISPLAY_STEP == 0 and i:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
            print("training accuracy:\t %.4f for step %d"%(train_accuracy, i))
        
        # train on batch
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
    
    
    # check final accuracy on validation set  
    if(VALIDATION_SIZE):
        print('validation accuracy:\t %.4f'%sess.run(accuracy, feed_dict={x: validation_images, y_: validation_labels, keep_prob: 1.0}))
    
 
    # predict test set
    predicted_lables = sess.run(predict, feed_dict={x: test_images, keep_prob: 1.0})
    
    # save results
    np.savetxt('submission_'+METHOD_NAME+'.csv', np.c_[range(1,len(test_images)+1),predicted_lables], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
        
    print("predicted_lables({0})".format(len(predicted_lables)))   

# <codecell>

# display third test image
display(test_images[2])
print (predicted_lables[2])

# <codecell>


