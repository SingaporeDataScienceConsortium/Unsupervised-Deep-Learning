# The code is shared on SDSC Github
# Description:
# use Autoencoder to encode the English letter, A

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

all_imgs = np.loadtxt('imgvectors.txt')

# 80% for training and 20% for testing
all_imgs_train = all_imgs[:2400,:] # for training
all_imgs_test = all_imgs[2400:,:]  # for testing

print("Data loaded !")

# basic arameters
learning_rate = 0.01
training_epochs = 100
batch_size = 128
display_step = 10
examples_to_show = 10 # show 10 pair of original and decoded images

# number of inputs = number of pixels in an image
n_input = all_imgs.shape[1]

# input to the encoder
X = tf.placeholder("float", [None, n_input]) # None -> batch size -> 128

# hidden layer settings
n_hidden_1 = 256 # 1st layer: number of hidden units
n_hidden_2 = 128 # 2nd layer: number of hidden units
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), # size = [784  256]
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), # size = [256  128]
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])), # size = [128  256]
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])), # size = [256  784]
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])), # size = [256]
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), # size = [128]
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])), # size = [256]
    'decoder_b2': tf.Variable(tf.random_normal([n_input])), # size = [784]
}

# Building the encoder
def encoder(x):

    # [128  784] * [784  256]
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))

    # [128  256] * [256  128]
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))

    return layer_2


# Building the decoder
def decoder(x):

    # [128  128] * [128  256]
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))

    # [128  256] * [256  784]
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))


    return layer_2

# Construct model: encoder and decoder
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op # Prediction
y_true = X # Targets (Labels)

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# randomly choose some images to monitor the training
img_show = all_imgs_test[np.random.randint(0,all_imgs_test.shape[0],examples_to_show),:]

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(all_imgs_train.shape[0]/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_img = all_imgs_train[i*batch_size:(i+1)*batch_size,:]
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_img})

        # display the progress
        if epoch % display_step == 0:
            print("Epoch:", '%d/%d' % (epoch,training_epochs), "cost=", "{:.9f}".format(c))

#        # regularly check performance
#        if epoch % display_step == 0:
#            plt.close("all")
#            encode_decode = sess.run(y_pred, feed_dict={X: img_show})
#            f, a = plt.subplots(2, 10, figsize=(10, 2))
#            for i in range(examples_to_show):
#                a[0][i].imshow(np.reshape(img_show[i], (28, 28)))
#                a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
#
#                a[0][i].get_xaxis().set_visible(False)
#                a[0][i].get_yaxis().set_visible(False)
#                a[1][i].get_xaxis().set_visible(False)
#                a[1][i].get_yaxis().set_visible(False)
#            plt.show()
#            plt.pause(0.5)

    print("Optimization Finished!")

    # show pairs of orginal (top row) and decoded (bottom row) images in the end
    encode_decode = sess.run(y_pred, feed_dict={X: img_show})
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(img_show[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

        a[0][i].get_xaxis().set_visible(False)
        a[0][i].get_yaxis().set_visible(False)
        a[1][i].get_xaxis().set_visible(False)
        a[1][i].get_yaxis().set_visible(False)
    plt.show()

