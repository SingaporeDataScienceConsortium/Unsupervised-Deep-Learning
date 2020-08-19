# The code is shared on SDSC Github
# Description:
# use Autoencoder to denoise images of English letters A, B, C, D and E

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

hidden_units = 32
noise_factor = 0.5 # maximum amplitude of noise
epochs = 50
batch_size = 128

all_imgs = np.loadtxt('imgvectors.txt')

# 80% for training and 20% for testing
all_imgs_train = all_imgs[:4000,:]
all_imgs_test = all_imgs[4000:,:]

Nimage = all_imgs_train.shape[0] # number of images/datasets
image_pixels = all_imgs_train.shape[1] # number of pixels in an image

# Input
inputs = tf.placeholder(tf.float32, [None, image_pixels]) # None = batch_size
targets = tf.placeholder(tf.float32, [None, image_pixels]) # None = batch_size

# construct the model

# 1st layer: [128  784] * [784  32]
hidden_layer = tf.layers.dense(inputs, hidden_units, activation=tf.nn.relu)

# 2nd layer: [128  32] * [32 784]
logits = tf.layers.dense(hidden_layer, image_pixels, activation=None)

# output: [128  784]
outputs = tf.sigmoid(logits)

# loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)
# target: expected result
# logits: predicted result
# noted that 'sigmoid_cross_entropy_with_logits' should be applied to the result before activation (sigmoid)

cost = tf.reduce_mean(loss)

# optimizer
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(Nimage//batch_size):
        imgs = all_imgs_train[ii*batch_size:(ii+1)*batch_size,:] # get a new batch

        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # '*imgs.shape' is used to unpack 'imgs.shape' which is a tuple

        noisy_imgs = np.clip(noisy_imgs, 0., 1.) # make sure image intensities are in the range of 0-1
        # anything <0 -> turn to 0
        # anything >1 -> turn to 1

        batch_cost, _ = sess.run([cost, optimizer], feed_dict={inputs: noisy_imgs, targets: imgs})
    if e%10==0:
        print("Epoch: {}/{}...".format(e, epochs),"Training loss: {:.4f}".format(batch_cost))

# define the configuration of the display
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))

# prepare noisy images made from test images
noisy_imgs = all_imgs_test + noise_factor * np.random.randn(*all_imgs_test.shape)
noisy_imgs = np.clip(noisy_imgs, 0.0, 1.0)

reconstructed = sess.run(outputs, feed_dict={inputs: noisy_imgs})

for images, row in zip([noisy_imgs, reconstructed], axes): # go through two rows, noisy and reconstructed image respectively
    for img, ax in zip(images, row): # go through each example
        ax.imshow(img.reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

sess.close()

