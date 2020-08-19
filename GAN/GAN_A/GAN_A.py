# The code is shared on SDSC Github
# Description:
# generate English letter A

import numpy as np
import tensorflow as tf
import os
tf.reset_default_graph() # remove old information and reset the training

g_hidden_size = 256 # for generator
d_hidden_size = 256 # for discriminator
alpha = 0.01 # used for the activation function in generator and discriminator

batch_size = 128
epoches = 100

learning_rate = 0.002

# Size of input image to discriminator
input_size = 784 # 28 * 28

# Size of latent vector to genorator
z_size = 100

# Smoothing
smooth = 0.1
# used for label smoothing; used with classifier
# for labels of 'real', we want all to be one. however, the real calculation is not ideal.
# to help the discriminator to generate better result, we need to slightly reduce the values of 'real' labels

datas = np.float32(np.loadtxt('imgvectors.txt'))
# split the original data into multiple batches
def batch_data(datas, batch_size):
    batches = []
    for i in range(datas.shape[0] // batch_size):
        batch = datas[i*batch_size : (i+1) * batch_size,:]
        batches.append(batch)
    return batches

batches = batch_data(datas,batch_size) # prepare data as batches

# define the inputs to the model
def model_inputs(real_dim, z_dim): # (784,100)
    inputs_real = tf.placeholder(tf.float32,(None, real_dim))
    inputs_z = tf.placeholder(tf.float32, (None, z_dim))
    return inputs_real, inputs_z

# Creat input placeholders
input_real, input_z = model_inputs(input_size, z_size) # (784,100)

# define the generator
def generator(z, out_dim, n_units=128, reuse = False, alpha = 0.01):
    with tf.variable_scope('generator', reuse = reuse):
        h1 = tf.layers.dense(z, n_units, activation = None)# Leaky ReLUense( z, n_units, activation = None )

        # LeakyRelu is usually used here as the activation function
        h1 = tf.maximum(alpha * h1, h1)
        # LeakyRelu is used to allow the gradient to flow backwards through the layer unimpeded

        logits = tf.layers.dense(h1, out_dim, activation = None)
        out = tf.tanh(logits) # tanh is recommended for generator
        return out

# Build the model
g_model = generator(input_z,input_size,n_units=g_hidden_size,alpha=alpha)
# g_model is the generator output

# define the discriminator
def discriminator(x, n_units = 128, reuse = False, alpha = 0.01):
    with tf.variable_scope('discriminator', reuse = reuse):
        h1 = tf.layers.dense(x, n_units, activation = None)
        h1 = tf.maximum(alpha * h1, h1)
        logits = tf.layers.dense(h1, 1, activation = None) # the output only has 1 dimension which indicates 'fake' or 'real'
        out = tf.sigmoid(logits) # sigmoid is recommended for discriminator
        return out, logits

d_model_real, d_logits_real = discriminator(input_real, n_units=d_hidden_size, alpha=alpha) # for real data
d_model_fake, d_logits_fake = discriminator(g_model, reuse =True, n_units=d_hidden_size, alpha=alpha ) # for fake data

# Calculate losses
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real,
                                                                     labels = tf.ones_like(d_logits_real)*(1-smooth)))
# tf.ones_like -> create a tensor with all elements set to 1
d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake,
                                                                      labels = tf.zeros_like(d_logits_fake)))
# tf.zeros_like -> create a tensor with all elements set to 0
d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake,
                                                                labels = tf.ones_like( d_logits_fake)))

# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables() # return a list of variables to be trained

# separate variables belonging to discriminator and generator respectively
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list = d_vars) # var_list -> variables to train
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list = g_vars) # var_list -> variables to train

# create a folder to store the results
if not( os.path.exists('generate')):
    os.makedirs('generate')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epoches):
        for batch in batches:
            batch_images = batch
            batch_images = batch_images*2-1 # normalize to -1 to 1

            # Sample random noise for G [128  100]
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict = {input_real:batch_images,input_z:batch_z})
            _ = sess.run(g_train_opt, feed_dict = {input_z:batch_z})

        print('epoch ',e,'.')

        # monitor the training by save images generated in different epoches
        # each time, we generate 5 images from 5 latent vectors
        # each time the 5 latent vectors are randomly generated
        sample_z = np.random.uniform(-1, 1, size=(5, z_size))

        # use the trained generator to generate images
        gen_samples = sess.run(generator(input_z, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha),
                               feed_dict={input_z:sample_z})

        # convert the data to real image size and vertically stack the 5 images

        gen_image = gen_samples.reshape((28*5, 28, 1))

        # convert image intensities to the range of 0-255 for display purpose
        gen_image = tf.cast(np.multiply( gen_image, 255), tf.uint8)

        # save the images to monitor the training
        with open('generate\epoch' + str(e) + '.jpg', 'wb') as img:
            img.write(sess.run(tf.image.encode_jpeg( gen_image)))
