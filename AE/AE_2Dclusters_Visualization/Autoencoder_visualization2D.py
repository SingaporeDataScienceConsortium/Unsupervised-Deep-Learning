# The code is shared on SDSC Github
# Description:
# use Autoencoder to visualize classified spatial points with x and y coordinates, in 2D

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# display option 1
# data arranged in order
data = np.loadtxt('2D_Clusters_order.txt')

## display option 2
## data arranged in random order
#data = np.loadtxt('2D_Clusters.txt')

print("Data loaded !")

# Basic parameters
learning_rate = 0.001
training_epochs = 200
batch_size = 128
display_step = 10

# number of inputs for each dataset
# here each dataset has two inputs
n_input = data.shape[1]

X = tf.placeholder("float", [None, n_input])

# hidden layers for encoder and decoder
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)), # size: [2  128]
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)), # size: [128  64]
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)), # size: [64  10]
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)), # size: [10  2]
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)), # size: [2  10]
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)), # size: [10  64]
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)), # size: [64  128]
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)), # size: [128  2]
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])), # size: [128]
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), # size: [64]
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])), # size: [10]
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])), # size: [2]
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])), # size: [10]
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), # size: [64]
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])), # size: [128]
    'decoder_b4': tf.Variable(tf.random_normal([n_input])), # size: [2]
}

def encoder(x):

    # [128  2] * [2  128]
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))

    # [128  128] * [128  64]
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))

    # [128  64] * [64  10]
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),biases['encoder_b3']))

    # [128  10] * [10  2]
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),biases['encoder_b4'])
    # here we don't add an activation function since we want to check the encoded information without being activated by activation functions
    # However, you can try adding an activation here. The plot is still able to show the distribution of the features of data

    return layer_4

def decoder(x):

    # [128  2] * [2  10]
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))

    # [128  10] * [10  64]
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))

    # [128  64] * [64  128]
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),biases['decoder_b3']))

    # [128  128] * [128  2]
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),biases['decoder_b4']))

    return layer_4

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# the following two parameters will be compared for optimization
y_pred = decoder_op # Prediction
y_true = X # Targets (Labels) are the input data

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(data.shape[0]/batch_size)

    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_data = data[i*batch_size:(i+1)*batch_size,:]
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_data})

        # show progress
        if epoch % display_step == 0:
            print("Epoch:", '%d/%d' % (epoch,training_epochs), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # use the trained encoder for feature extraction
    encoder_result = sess.run(encoder_op, feed_dict={X: data})

    fig = plt.figure() # initialize a figure to visualize the result

    # display option 1
    # show clustered result with color-coded labels
    plt.scatter(encoder_result[0:1000, 0], encoder_result[0:1000, 1], c='r')
    plt.scatter(encoder_result[1000:2000, 0], encoder_result[1000:2000, 1], c='g')
    plt.scatter(encoder_result[2000:3000, 0], encoder_result[2000:3000, 1], c='b')
    plt.scatter(encoder_result[3000:4000, 0], encoder_result[3000:4000, 1], c='k')

#    # display option 2
#    # show clustered result
#    plt.scatter(encoder_result[:,0], encoder_result[:,1], c='b')

    plt.show()